param(
  [string]$FrameDir = "c:\Users\mtus6\RenjuDQN\images\cnn-kernel-frames",
  [string]$OutFile  = "c:\Users\mtus6\RenjuDQN\images\cnn-kernel.gif",
  [int]$FrameCount = 9,
  [int]$DelayCentis = 70,
  [int]$LastDelayCentis = 180
)

Add-Type -AssemblyName System.Drawing

$csharp = @'
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class GifTool
{
    public static Color[] BuildPalette(string[] samplePngs, int maxColors)
    {
        var counts = new Dictionary<int, int>();
        foreach (var path in samplePngs)
        {
            using (var bmp = new Bitmap(path))
            {
                var data = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                byte[] bytes = new byte[data.Stride * bmp.Height];
                Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
                bmp.UnlockBits(data);
                int stride = data.Stride;
                for (int y = 0; y < bmp.Height; y++)
                {
                    int row = y * stride;
                    for (int x = 0; x < bmp.Width; x++)
                    {
                        int i = row + x * 4;
                        int b = bytes[i], g = bytes[i + 1], r = bytes[i + 2];
                        int key = (r << 16) | (g << 8) | b;
                        int cur;
                        counts[key] = counts.TryGetValue(key, out cur) ? cur + 1 : 1;
                    }
                }
            }
        }

        return counts.OrderByDescending(kv => kv.Value)
                      .Take(maxColors)
                      .Select(kv => Color.FromArgb((kv.Key >> 16) & 0xFF, (kv.Key >> 8) & 0xFF, kv.Key & 0xFF))
                      .ToArray();
    }

    public static Bitmap Quantize(Bitmap src, Color[] palette)
    {
        int w = src.Width, h = src.Height;
        var srcData = src.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        int srcStride = srcData.Stride;
        byte[] srcBytes = new byte[srcStride * h];
        Marshal.Copy(srcData.Scan0, srcBytes, 0, srcBytes.Length);
        src.UnlockBits(srcData);

        var dst = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
        var pal = dst.Palette;
        for (int i = 0; i < 256; i++) pal.Entries[i] = i < palette.Length ? palette[i] : Color.Black;
        dst.Palette = pal;

        var dstData = dst.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
        int dstStride = dstData.Stride;
        byte[] dstBytes = new byte[dstStride * h];

        int n = palette.Length;
        int[] pr = new int[n], pg = new int[n], pb = new int[n];
        for (int i = 0; i < n; i++) { pr[i] = palette[i].R; pg[i] = palette[i].G; pb[i] = palette[i].B; }

        for (int y = 0; y < h; y++)
        {
            int srowOff = y * srcStride;
            int drowOff = y * dstStride;
            for (int x = 0; x < w; x++)
            {
                int b = srcBytes[srowOff + x * 4 + 0];
                int g = srcBytes[srowOff + x * 4 + 1];
                int r = srcBytes[srowOff + x * 4 + 2];
                int best = 0, bestDist = int.MaxValue;
                for (int i = 0; i < n; i++)
                {
                    int dr = r - pr[i], dg = g - pg[i], db = b - pb[i];
                    int dist = dr * dr + dg * dg + db * db;
                    if (dist < bestDist) { bestDist = dist; best = i; }
                }
                dstBytes[drowOff + x] = (byte)best;
            }
        }
        Marshal.Copy(dstBytes, 0, dstData.Scan0, dstBytes.Length);
        dst.UnlockBits(dstData);
        return dst;
    }

    public static byte[] ExtractBlock(Bitmap quantized, out int colorTableSizeBits)
    {
        using (var ms = new MemoryStream())
        {
            quantized.Save(ms, ImageFormat.Gif);
            byte[] data = ms.ToArray();

            byte packedLSD = data[10];
            bool gctFlag = (packedLSD & 0x80) != 0;
            int gctSizeBits = packedLSD & 0x07;
            int gctSize = gctFlag ? (int)Math.Pow(2, gctSizeBits + 1) * 3 : 0;

            int pos = 13;
            byte[] colorTable = new byte[gctSize];
            Array.Copy(data, pos, colorTable, 0, gctSize);
            pos += gctSize;

            while (data[pos] != 0x2C)
            {
                if (data[pos] == 0x21)
                {
                    pos += 2;
                    while (data[pos] != 0x00)
                    {
                        int blockSize = data[pos];
                        pos += 1 + blockSize;
                    }
                    pos += 1;
                }
                else
                {
                    throw new Exception("Unexpected byte 0x" + data[pos].ToString("X2") + " at " + pos);
                }
            }

            int imgDescStart = pos;
            int trailerPos = data.Length - 1;
            while (data[trailerPos] != 0x3B) trailerPos--;

            int len = trailerPos - imgDescStart;
            byte[] block = new byte[10 + colorTable.Length + (len - 10)];

            Array.Copy(data, imgDescStart, block, 0, 10);
            block[9] = (byte)(0x80 | (gctSizeBits & 0x07));
            Array.Copy(colorTable, 0, block, 10, colorTable.Length);
            Array.Copy(data, imgDescStart + 10, block, 10 + colorTable.Length, len - 10);

            colorTableSizeBits = gctSizeBits;
            return block;
        }
    }
}
'@

Add-Type -TypeDefinition $csharp -ReferencedAssemblies System.Drawing

$samplePngs = @(0..($FrameCount-1)) | ForEach-Object { Join-Path $FrameDir "frame_$_.png" }
$palette = [GifTool]::BuildPalette($samplePngs, 250)
Write-Output "palette size: $($palette.Length)"

$out = New-Object System.IO.MemoryStream
function W([byte[]]$b) { $out.Write($b, 0, $b.Length) }

W([System.Text.Encoding]::ASCII.GetBytes("GIF89a"))

$width = 1000; $height = 400
W([byte[]]@(($width -band 0xFF), (($width -shr 8) -band 0xFF)))
W([byte[]]@(($height -band 0xFF), (($height -shr 8) -band 0xFF)))
W([byte[]]@(0x00, 0x00, 0x00))

W([byte[]]@(0x21, 0xFF, 0x0B))
W([System.Text.Encoding]::ASCII.GetBytes("NETSCAPE2.0"))
W([byte[]]@(0x03, 0x01, 0x00, 0x00, 0x00))

for ($k = 0; $k -lt $FrameCount; $k++) {
  $png = Join-Path $FrameDir "frame_$k.png"
  $orig = New-Object System.Drawing.Bitmap($png)
  $q = [GifTool]::Quantize($orig, $palette)
  $orig.Dispose()

  $bits = 0
  $block = [GifTool]::ExtractBlock($q, [ref]$bits)
  $q.Dispose()

  $delay = if ($k -eq $FrameCount - 1) { $LastDelayCentis } else { $DelayCentis }
  $delayLo = $delay -band 0xFF
  $delayHi = ($delay -shr 8) -band 0xFF
  W([byte[]]@(0x21, 0xF9, 0x04, 0x04, $delayLo, $delayHi, 0x00, 0x00))

  W($block)
}

W([byte[]]@(0x3B))

[System.IO.File]::WriteAllBytes($OutFile, $out.ToArray())
Write-Output "wrote $OutFile ($($out.Length) bytes)"
