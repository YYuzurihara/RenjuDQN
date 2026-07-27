#!/usr/bin/env bash
set -e
OUT="$(dirname "$0")"

for r in 0 1 2; do
  for c in 0 1 2; do
    k=$((r*3+c))
    f="$OUT/frame_$k.svg"

    {
      echo '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 400" font-family="Helvetica, Arial, sans-serif">'
      echo '  <rect x="0" y="0" width="1000" height="400" fill="#ffffff"/>'
      echo '  <defs>'
      echo '    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
      echo '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333333"/>'
      echo '    </marker>'
      echo '  </defs>'

      # input grid 5x5, cell 70, origin (60,30)
      for i in 0 1 2 3 4; do
        for j in 0 1 2 3 4; do
          x=$((60+j*70))
          y=$((30+i*70))
          echo "  <rect x=\"$x\" y=\"$y\" width=\"70\" height=\"70\" fill=\"#dae8fc\" stroke=\"#6c8ebf\" stroke-width=\"2\"/>"
        done
      done

      # kernel highlight overlay (3x3 block at r,c)
      kx=$((60+c*70))
      ky=$((30+r*70))
      echo "  <rect x=\"$kx\" y=\"$ky\" width=\"210\" height=\"210\" fill=\"#f8cecc\" fill-opacity=\"0.55\" stroke=\"#b85450\" stroke-width=\"6\"/>"

      # output grid 3x3, cell 80, origin (650,30)
      for i in 0 1 2; do
        for j in 0 1 2; do
          x=$((650+j*80))
          y=$((30+i*80))
          idx=$((i*3+j))
          if [ "$idx" -lt "$k" ]; then
            fill="#d5e8d4"; stroke="#82b366"
          elif [ "$idx" -eq "$k" ]; then
            fill="#f8cecc"; stroke="#b85450"
          else
            fill="#f5f5f5"; stroke="#cccccc"
          fi
          echo "  <rect x=\"$x\" y=\"$y\" width=\"80\" height=\"80\" fill=\"$fill\" stroke=\"$stroke\" stroke-width=\"3\"/>"
        done
      done

      # arrow from kernel center to current output cell center
      kcx=$((kx+105))
      kcy=$((ky+105))
      ocx=$((650+c*80+40))
      ocy=$((30+r*80+40))
      echo "  <line x1=\"$kcx\" y1=\"$kcy\" x2=\"$((ocx-45))\" y2=\"$ocy\" stroke=\"#333333\" stroke-width=\"3\" marker-end=\"url(#arrow)\"/>"

      echo '</svg>'
    } > "$f"
  done
done
echo "generated 9 frames"
