export const DEFAULT_METADATA = Object.freeze({
  input_name: "state",
  output_name: "q_values",
  input_dtype: "float32",
  board_size: 15,
  board_cells: 225,
  in_channels: 4,
  num_move_labels: 225,
  supports_batch: false,
});

let ortRuntime = globalThis.ort ?? null;

export function setOrtRuntime(runtime) {
  ortRuntime = runtime;
}

function getOrtRuntime() {
  const runtime = ortRuntime ?? globalThis.ort ?? null;
  if (!runtime) {
    throw new Error(
      "ONNX Runtime Web is not available. Load ort.min.js first or call setOrtRuntime(ort).",
    );
  }
  return runtime;
}

export async function createSession(modelUrl, sessionOptions = {}) {
  const resolvedOptions = {
    executionProviders: ["wasm"],
    ...sessionOptions,
  };
  return getOrtRuntime().InferenceSession.create(modelUrl, resolvedOptions);
}

export async function loadMetadata(metadataUrl) {
  const response = await fetch(metadataUrl);
  if (!response.ok) {
    throw new Error(`Failed to load metadata: ${response.status} ${response.statusText}`);
  }
  const metadata = await response.json();
  return {...DEFAULT_METADATA, ...metadata};
}

export function parseBoardCsv(boardCsv) {
  return boardCsv
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
    .map((value) => Number.parseInt(value, 10));
}

export function validateBoard(board, metadata = DEFAULT_METADATA) {
  if (!Array.isArray(board)) {
    throw new TypeError("Board must be an array of integers.");
  }
  if (board.length !== metadata.board_cells) {
    throw new Error(`Expected ${metadata.board_cells} board cells, got ${board.length}.`);
  }
  for (const cell of board) {
    if (cell !== 0 && cell !== 1 && cell !== 2) {
      throw new Error(`Board contains invalid cell value: ${cell}`);
    }
  }
}

// Mirrors src/renju_dqn/board_encoder.py::encode_board -- channel order (own stones /
// opponent stones / last move / side-to-move constant) must match the trained model.
export function encodeBoardTensor(board, player, lastMoveIndex, metadata = DEFAULT_METADATA) {
  validateBoard(board, metadata);
  const cells = metadata.board_cells;
  const opponent = player === BLACK ? WHITE : BLACK;
  const data = new Float32Array(metadata.in_channels * cells);
  const ownOffset = 0;
  const opponentOffset = cells;
  const lastMoveOffset = 2 * cells;
  const sideOffset = 3 * cells;
  const sideValue = player === BLACK ? 1 : 0;

  for (let index = 0; index < cells; index += 1) {
    data[ownOffset + index] = board[index] === player ? 1 : 0;
    data[opponentOffset + index] = board[index] === opponent ? 1 : 0;
    data[sideOffset + index] = sideValue;
  }
  if (lastMoveIndex !== null && lastMoveIndex !== undefined) {
    data[lastMoveOffset + lastMoveIndex] = 1;
  }
  return data;
}

function idxToRowCol(index, boardSize) {
  return [Math.floor(index / boardSize), index % boardSize];
}

function rowColToIdx(row, col, boardSize) {
  return row * boardSize + col;
}

function inside(row, col, boardSize) {
  return row >= 0 && row < boardSize && col >= 0 && col < boardSize;
}

function boardWithMove(board, index, player) {
  const nextBoard = board.slice();
  nextBoard[index] = player;
  return nextBoard;
}

function stoneCounts(board) {
  let blackCount = 0;
  let whiteCount = 0;
  for (const cell of board) {
    if (cell === 1) {
      blackCount += 1;
    } else if (cell === 2) {
      whiteCount += 1;
    }
  }
  return [blackCount, whiteCount];
}

function contiguousCount(board, index, player, dr, dc, boardSize) {
  let total = 1;
  const [row, col] = idxToRowCol(index, boardSize);

  let step = 1;
  while (inside(row + dr * step, col + dc * step, boardSize)) {
    if (board[rowColToIdx(row + dr * step, col + dc * step, boardSize)] !== player) {
      break;
    }
    total += 1;
    step += 1;
  }

  step = 1;
  while (inside(row - dr * step, col - dc * step, boardSize)) {
    if (board[rowColToIdx(row - dr * step, col - dc * step, boardSize)] !== player) {
      break;
    }
    total += 1;
    step += 1;
  }

  return total;
}

function hasFiveOrMore(board, index, player, boardSize) {
  return DIRECTIONS.some(([dr, dc]) => contiguousCount(board, index, player, dr, dc, boardSize) >= 5);
}

function isOverline(board, index, player, boardSize) {
  return DIRECTIONS.some(([dr, dc]) => contiguousCount(board, index, player, dr, dc, boardSize) >= 6);
}

function linePointsThrough(index, dr, dc, boardSize) {
  let [row, col] = idxToRowCol(index, boardSize);
  while (inside(row - dr, col - dc, boardSize)) {
    row -= dr;
    col -= dc;
  }

  const points = [];
  while (inside(row, col, boardSize)) {
    points.push(rowColToIdx(row, col, boardSize));
    row += dr;
    col += dc;
  }
  return points;
}

function immediateWinsInDirection(board, player, linePoints, boardSize) {
  const wins = new Set();
  for (const candidate of linePoints) {
    if (board[candidate] !== EMPTY) {
      continue;
    }
    const nextBoard = boardWithMove(board, candidate, player);
    if (player === BLACK && isOverline(nextBoard, candidate, BLACK, boardSize)) {
      continue;
    }
    if (hasFiveOrMore(nextBoard, candidate, player, boardSize)) {
      wins.add(candidate);
    }
  }
  return wins;
}

function countFourDirections(board, move, player, boardSize) {
  let count = 0;
  for (const [dr, dc] of DIRECTIONS) {
    const linePoints = linePointsThrough(move, dr, dc, boardSize);
    if (immediateWinsInDirection(board, player, linePoints, boardSize).size > 0) {
      count += 1;
    }
  }
  return count;
}

function countOpenThreeDirections(board, move, player, boardSize) {
  let count = 0;
  for (const [dr, dc] of DIRECTIONS) {
    const linePoints = linePointsThrough(move, dr, dc, boardSize);
    let foundOpenThree = false;
    for (const candidate of linePoints) {
      if (board[candidate] !== EMPTY) {
        continue;
      }
      const nextBoard = boardWithMove(board, candidate, player);
      if (player === BLACK && isOverline(nextBoard, candidate, BLACK, boardSize)) {
        continue;
      }
      const winningPoints = immediateWinsInDirection(nextBoard, player, linePoints, boardSize);
      if (winningPoints.size >= 2) {
        foundOpenThree = true;
        break;
      }
    }
    if (foundOpenThree) {
      count += 1;
    }
  }
  return count;
}

export function inferPlayer(board) {
  const [blackCount, whiteCount] = stoneCounts(board);
  if (blackCount === whiteCount) {
    return BLACK;
  }
  if (blackCount === whiteCount + 1) {
    return WHITE;
  }
  throw new Error(
    `Invalid board: black_count=${blackCount}, white_count=${whiteCount}. ` +
      "Expected black == white or black == white + 1.",
  );
}

function isForbiddenForBlack(board, index, boardSize) {
  if (board[index] !== EMPTY) {
    return true;
  }

  const [blackCount, whiteCount] = stoneCounts(board);
  const moveNumber = blackCount + whiteCount;
  const centerIndex = Math.floor(boardSize / 2) * boardSize + Math.floor(boardSize / 2);
  if (moveNumber === 0) {
    return index !== centerIndex;
  }

  const nextBoard = boardWithMove(board, index, BLACK);
  if (isOverline(nextBoard, index, BLACK, boardSize)) {
    return true;
  }
  if (countFourDirections(nextBoard, index, BLACK, boardSize) >= 2) {
    return true;
  }
  if (countOpenThreeDirections(nextBoard, index, BLACK, boardSize) >= 2) {
    return true;
  }
  return false;
}

export function legalMoveMask(board, metadata = DEFAULT_METADATA) {
  validateBoard(board, metadata);
  const player = inferPlayer(board);
  const mask = new Array(metadata.board_cells);
  for (let index = 0; index < metadata.board_cells; index += 1) {
    if (board[index] !== EMPTY) {
      mask[index] = false;
      continue;
    }
    mask[index] = player === BLACK ? !isForbiddenForBlack(board, index, metadata.board_size) : true;
  }
  return mask;
}

export function boardWinner(board, metadata = DEFAULT_METADATA) {
  validateBoard(board, metadata);
  for (let index = 0; index < board.length; index += 1) {
    if (board[index] === BLACK && isOverline(board, index, BLACK, metadata.board_size)) {
      return WHITE;
    }
  }
  for (let index = 0; index < board.length; index += 1) {
    if (board[index] === BLACK && hasFiveOrMore(board, index, BLACK, metadata.board_size)) {
      return BLACK;
    }
  }
  for (let index = 0; index < board.length; index += 1) {
    if (board[index] === WHITE && hasFiveOrMore(board, index, WHITE, metadata.board_size)) {
      return WHITE;
    }
  }
  return null;
}

export function boardIsFull(board, metadata = DEFAULT_METADATA) {
  validateBoard(board, metadata);
  return board.every((cell) => cell !== EMPTY);
}

// Mirrors src/renju_dqn/reward.py — keep the two in sync.
export const DRAW = 0;
export const FOUR_WEIGHT = 9;
export const OPEN_THREE_WEIGHT = 3;
export const POTENTIAL_SCALE = 5.0;
export const DEFAULT_SHAPING_COEFFICIENT = 0.1;

export function stoneThreatScore(board, index, player, boardSize = DEFAULT_METADATA.board_size) {
  const fours = countFourDirections(board, index, player, boardSize);
  const openThrees = countOpenThreeDirections(board, index, player, boardSize);
  return FOUR_WEIGHT * fours + OPEN_THREE_WEIGHT * openThrees;
}

export function boardPotential(board, player, boardSize = DEFAULT_METADATA.board_size) {
  const opponent = player === BLACK ? WHITE : BLACK;
  let ownScore = 0;
  let opponentScore = 0;
  for (let index = 0; index < board.length; index += 1) {
    const cell = board[index];
    if (cell === player) {
      ownScore += stoneThreatScore(board, index, player, boardSize);
    } else if (cell === opponent) {
      opponentScore += stoneThreatScore(board, index, opponent, boardSize);
    }
  }
  return ownScore - opponentScore;
}

export function normalizedPotential(
  board,
  player,
  scale = POTENTIAL_SCALE,
  boardSize = DEFAULT_METADATA.board_size,
) {
  return Math.tanh(boardPotential(board, player, boardSize) / scale);
}

export function terminalReward(player, winner) {
  if (winner === DRAW) {
    return 0.0;
  }
  return winner === player ? 1.0 : -1.0;
}

export function computeReward(
  board,
  nextBoard,
  player,
  winner,
  done,
  gamma,
  coefficient = DEFAULT_SHAPING_COEFFICIENT,
  scale = POTENTIAL_SCALE,
  boardSize = DEFAULT_METADATA.board_size,
) {
  const sparse = done ? terminalReward(player, winner) : 0.0;
  const phiT = normalizedPotential(board, player, scale, boardSize);
  const phiNext = done ? 0.0 : normalizedPotential(nextBoard, player, scale, boardSize);
  return sparse + coefficient * (gamma * phiNext - phiT);
}

export function playerLabel(player) {
  if (player === BLACK) {
    return "black";
  }
  if (player === WHITE) {
    return "white";
  }
  return "unknown";
}

export {BLACK, EMPTY, WHITE};

function topK(values, k) {
  return values
    .map((value, index) => ({index, value}))
    .sort((left, right) => right.value - left.value)
    .slice(0, k);
}

export async function predictBoard(
  session,
  board,
  player,
  lastMoveIndex,
  metadata = DEFAULT_METADATA,
  options = {},
) {
  const ort = getOrtRuntime();
  const resolvedMetadata = {...DEFAULT_METADATA, ...metadata};
  validateBoard(board, resolvedMetadata);

  const tensorData = encodeBoardTensor(board, player, lastMoveIndex, resolvedMetadata);
  const inputTensor = new ort.Tensor(resolvedMetadata.input_dtype, tensorData, [
    1,
    resolvedMetadata.in_channels,
    resolvedMetadata.board_size,
    resolvedMetadata.board_size,
  ]);
  const inputName = resolvedMetadata.input_name ?? session.inputNames[0];
  const outputName = resolvedMetadata.output_name ?? session.outputNames[0];
  const outputs = await session.run({[inputName]: inputTensor});
  const qValuesTensor = outputs[outputName];
  if (!qValuesTensor) {
    throw new Error(`Model output "${outputName}" was not found.`);
  }

  const qValues = Array.from(qValuesTensor.data, (value) => Number(value));
  const applyLegalMask = options.applyLegalMask ?? true;
  const maskedQValues = qValues.slice();
  let mask = null;
  if (applyLegalMask) {
    mask = legalMoveMask(board, resolvedMetadata);
    if (!mask.some(Boolean)) {
      throw new Error("No legal moves available for the provided board.");
    }
    for (let index = 0; index < maskedQValues.length; index += 1) {
      if (!mask[index]) {
        maskedQValues[index] = Number.NEGATIVE_INFINITY;
      }
    }
  }

  const topKCount = Math.min(options.topK ?? 5, maskedQValues.length);
  const ranked = topK(maskedQValues, topKCount).map((item, rank) => ({
    rank: rank + 1,
    index: item.index,
    qValue: item.value,
  }));

  return {
    qValues,
    maskedQValues,
    legalMask: mask,
    predictedIndex: ranked[0].index,
    predictedQValue: ranked[0].qValue,
    topK: ranked,
  };
}

export async function predictBoardCsv(
  session,
  boardCsv,
  player,
  lastMoveIndex,
  metadata = DEFAULT_METADATA,
  options = {},
) {
  return predictBoard(session, parseBoardCsv(boardCsv), player, lastMoveIndex, metadata, options);
}

export async function loadModel(modelUrl, metadataUrl, sessionOptions = {}) {
  const [session, metadata] = await Promise.all([
    createSession(modelUrl, sessionOptions),
    loadMetadata(metadataUrl),
  ]);
  return {session, metadata};
}

const EMPTY = 0;
const BLACK = 1;
const WHITE = 2;
const DIRECTIONS = [
  [1, 0],
  [0, 1],
  [1, 1],
  [1, -1],
];
