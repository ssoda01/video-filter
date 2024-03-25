import { WorkerPool } from "./worker";
import {
  filter,
  return_pointer,
  take_pointer_by_value,
  filter_by_block,
} from "rust-filter/rust_filter";
import { memory } from "rust-filter/rust_filter_bg.wasm";

const workerPool = new WorkerPool(4);

export enum FilterOption {
  off = "off",
  js = "js",
  jsWebworker = "jsWebworker",
  wasmGo = "wasmGo",
  wasmRust = "wasmRust",
  wasmRustSharedMem = "wasmRustSharedMem",
  wasmRustWebworker = "wasmRustWebworker",
}

const filterTimeRecordsMap: { [k: string]: number[] } = {
  [FilterOption.off]: [],
  [FilterOption.js]: [],
  [FilterOption.jsWebworker]: [],
  [FilterOption.wasmGo]: [],
  [FilterOption.wasmRust]: [],
  [FilterOption.wasmRustSharedMem]: [],
  [FilterOption.wasmRustWebworker]: [],
};

export enum Kernel {
  sharpen = "sharpen",
  laplace = "laplace",
  smoothing = "smoothing",
}

const kernelMap = {
  [Kernel.smoothing]: [
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
  ],
  [Kernel.sharpen]: [
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1],
  ],
  [Kernel.laplace]: [
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1],
  ],
};

function calcFPS(arr: number[]) {
  const n = 20;
  if (arr.length > n) {
    arr.shift();
  }
  let averageTime =
    arr.reduce((pre, item) => {
      return pre + item;
    }, 0) / arr.length;
  return 1000 / averageTime;
}

function getDenominator(kernel: number[][]) {
  let sum = 0;
  for (let i = 0; i < kernel.length; i++) {
    const arr = kernel[i];
    for (let j = 0; j < arr.length; j++) {
      sum += Math.abs(kernel[i][j]);
    }
  }
  return sum;
}

function _filterByJS(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  kernel: number[][]
) {
  // const newData = new Uint8ClampedArray(data)
  const h = kernel.length,
    w = h;
  const half = Math.floor(h / 2);

  // picture iteration.
  for (let y = half; y < height - half; ++y) {
    for (let x = half; x < width - half; ++x) {
      const px = (y * width + x) * 4; // pixel index.
      let r = 0,
        g = 0,
        b = 0;

      // core iteration.
      for (let cy = 0; cy < h; ++cy) {
        for (let cx = 0; cx < w; ++cx) {
          // dealing edge case.
          const cpx = ((y + (cy - half)) * width + (x + (cx - half))) * 4;
          // if (px === 50216) debugger
          r += data[cpx + 0] * kernel[cy][cx];
          g += data[cpx + 1] * kernel[cy][cx];
          b += data[cpx + 2] * kernel[cy][cx];
          // a += data[cpx + 3] * kernel[cy][cx]
        }
      }
      data[px + 0] = r > 255 ? 255 : r < 0 ? 0 : r;
      data[px + 1] = g > 255 ? 255 : g < 0 ? 0 : g;
      data[px + 2] = b > 255 ? 255 : b < 0 ? 0 : b;
      // data[px + 3] =
      //   a / denominator > 255 ? 255 : a / denominator < 0 ? 0 : a / denominator
    }
  }
  return data;
}

function filterByJS(
  // function hdrFilterByJS(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  exposureTimes: number[],
  toneMappingAlgorithm: string
) {
  const numExposures = exposureTimes.length;
  const numPixels = width * height;
  const hdrData = new Float32Array(numPixels * 3);

  // Convert input image data to HDR format
  for (let i = 0; i < numPixels; i++) {
    hdrData[i * 3] = data[i * 4] / 255.0; // Red channel
    hdrData[i * 3 + 1] = data[i * 4 + 1] / 255.0; // Green channel
    hdrData[i * 3 + 2] = data[i * 4 + 2] / 255.0; // Blue channel
  }

  // Generate multiple exposure images
  const exposureImages = [];
  for (let i = 0; i < numExposures; i++) {
    const exposureImage = applyExposure(hdrData, exposureTimes[i]);
    exposureImages.push(exposureImage);
  }

  // Apply tone mapping algorithm to each exposure image
  const toneMappedImages = [];
  for (let i = 0; i < numExposures; i++) {
    const toneMappedImage = applyToneMapping(
      exposureImages[i],
      toneMappingAlgorithm
    );
    toneMappedImages.push(toneMappedImage);
  }

  // Blend tone-mapped images to create final HDR image
  const blendedImage = blendImages(toneMappedImages);

  // Convert HDR image back to 8-bit integer format
  for (let i = 0; i < numPixels; i++) {
    data[i * 4] = Math.round(blendedImage[i * 3] * 255); // Red channel
    data[i * 4 + 1] = Math.round(blendedImage[i * 3 + 1] * 255); // Green channel
    data[i * 4 + 2] = Math.round(blendedImage[i * 3 + 2] * 255); // Blue channel
  }

  return data;
}

// Helper function to apply exposure adjustment to HDR image
function applyExposure(hdrData: Float32Array, exposureTime: number) {
  const numPixels = hdrData.length / 3;
  const adjustedImage = new Float32Array(numPixels * 3);

  for (let i = 0; i < numPixels; i++) {
    adjustedImage[i * 3] = hdrData[i * 3] * exposureTime; // Red channel
    adjustedImage[i * 3 + 1] = hdrData[i * 3 + 1] * exposureTime; // Green channel
    adjustedImage[i * 3 + 2] = hdrData[i * 3 + 2] * exposureTime; // Blue channel
  }

  return adjustedImage;
}

// Helper function to apply tone mapping to HDR image
function applyToneMapping(hdrData: Float32Array, algorithm: string) {
  const numPixels = hdrData.length / 3;
  const toneMappedImage = new Float32Array(numPixels * 3);

  switch (algorithm) {
    case "reinhard":
      // Apply Reinhard tone mapping algorithm
      for (let i = 0; i < numPixels; i++) {
        const luminance = computeLuminance(
          hdrData[i * 3],
          hdrData[i * 3 + 1],
          hdrData[i * 3 + 2]
        );
        const toneMappedLuminance = reinhardToneMapping(luminance);
        const scaleFactor = toneMappedLuminance / luminance;

        toneMappedImage[i * 3] = hdrData[i * 3] * scaleFactor; // Red channel
        toneMappedImage[i * 3 + 1] = hdrData[i * 3 + 1] * scaleFactor; // Green channel
        toneMappedImage[i * 3 + 2] = hdrData[i * 3 + 2] * scaleFactor; // Blue channel
      }
      break;

    case "tmo":
      // Apply Tone Mapping Operator (TMO) algorithm
      // Implement the specific TMO algorithm of your choice
      break;

    // Add cases for other tone mapping algorithms

    default:
      // No tone mapping algorithm specified
      toneMappedImage.set(hdrData);
      break;
  }

  return toneMappedImage;
}
// Helper function to compute luminance from RGB values
function computeLuminance(red: number, green: number, blue: number) {
  return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}
// Helper function for Reinhard tone mapping algorithm
function reinhardToneMapping(luminance: number) {
  const key = 0.18; // Key value for the scene
  const white = 1.0; // White point

  return (luminance * (1 + luminance / (key * key))) / (1 + luminance);
}

// Helper function to blend multiple images
function blendImages(images: Float32Array[]) {
  const numPixels = images[0].length / 3;
  const blendedImage = new Float32Array(numPixels * 3);

  for (let i = 0; i < numPixels; i++) {
    let sumR = 0,
      sumG = 0,
      sumB = 0;
    for (let j = 0; j < images.length; j++) {
      sumR += images[j][i * 3];
      sumG += images[j][i * 3 + 1];
      sumB += images[j][i * 3 + 2];
    }
    blendedImage[i * 3] = sumR / images.length; // Red channel
    blendedImage[i * 3 + 1] = sumG / images.length; // Green channel
    blendedImage[i * 3 + 2] = sumB / images.length; // Blue channel
  }

  return blendedImage;
}

type Pointer = number;

function filterByGO(
  ptr: Pointer,
  width: number,
  height: number,
  kernel: number[][]
) {
  // @ts-ignore
  window.filterByGO(ptr, width, height, kernel);
}

async function filterByRust(
  ctxHidden: CanvasRenderingContext2D,
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  kernel: number[][]
) {
  filter(ctxHidden, ctx, width, height, new Float32Array([].concat(...kernel)));
}

function getSharedBuffer(imageData: Uint8ClampedArray) {
  const sharedArrayBuffer = new SharedArrayBuffer(imageData.buffer.byteLength);
  new Uint8ClampedArray(sharedArrayBuffer).set(imageData);
  return sharedArrayBuffer;
}

function getPtr(imageData: Uint8ClampedArray) {
  const ptr = return_pointer();
  // debugger
  // while (imageData.byteLength > memory.buffer.byteLength) {
  //   memory.grow(1)
  // }
  // debugger
  const uint8ClampedArray = new Uint8ClampedArray(memory.buffer);
  uint8ClampedArray.set(imageData);

  return ptr;
}
1114112;
1179648;
export function getDrawFn(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  canvasHidden: HTMLCanvasElement,
  afterEachFrame: (fps: number) => void,
  // useWebWorker: boolean = true,
  filterOption: FilterOption = FilterOption.off,
  kernel: Kernel = Kernel.sharpen
) {
  const context2D = canvas.getContext("2d")!;
  const context2DHidden = canvasHidden.getContext("2d")!;
  const size = canvas.height * canvas.width * 4;
  //@ts-ignore
  // For Go WASM
  // const {internalptr: ptr} = window.initShareMemory(size)
  // const mem = new Uint8ClampedArray(buffer, ptr, size)

  const draw = async () => {
    // record performance.
    const timeStart = performance.now();

    // render the first frame from the top-left of the canvas.
    context2DHidden.drawImage(
      video,
      0,
      0,
      video.videoWidth,
      video.videoHeight,
      0,
      0,
      canvas.width,
      canvas.height
    );

    context2DHidden.getImageData;

    // get current video data.
    const pixels = context2DHidden.getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    );

    switch (filterOption) {
      case FilterOption.js:
        pixels.data.set(
          filterByJS(
            pixels.data,
            canvas.width,
            canvas.height,
            [1.0, 2.0, 4.0],
            "reinhard"
          )
        );
        context2D.putImageData(pixels, 0, 0);
        break;
      case FilterOption.jsWebworker: {
        const sharedArrayBuffer = getSharedBuffer(pixels.data);
        await workerPool.filter({
          sharedArrayBuffer,
          width: canvas.width,
          height: canvas.height,
          kernel: kernelMap[kernel],
        });
        pixels.data.set(new Uint8ClampedArray(sharedArrayBuffer));
        context2D.putImageData(pixels, 0, 0);
        break;
      }

      case FilterOption.wasmRust:
        filterByRust(
          context2DHidden,
          context2D,
          canvas.width,
          canvas.height,
          kernelMap[kernel]
        );
        break;
      case FilterOption.wasmRustSharedMem:
        const ptr = getPtr(pixels.data);
        // memory.grow(1)
        filter_by_block(
          ptr,
          canvas.width,
          0,
          canvas.height,
          // new Float32Array([])
          new Float32Array([].concat(...kernelMap[kernel]))
        );
        pixels.data.set(
          new Uint8ClampedArray(memory.buffer).slice(
            0,
            canvas.width * canvas.height * 4
          )
        );
        context2D.putImageData(pixels, 0, 0);
        break;
      case FilterOption.wasmRustWebworker: {
        const sharedArrayBuffer = getSharedBuffer(pixels.data);
        // const sha
        // const shared = new Uint8ClampedArray(
        //   new SharedArrayBuffer(pixels.data.byteLength)
        // )
        // shared.set(new Uint8ClampedArray(memory.buffer))
        // const sharedArrayBuffer = new Uint8ClampedArray(mem)
        await workerPool.filter({
          useWasm: true,
          sharedArrayBuffer,
          width: canvas.width,
          height: canvas.height,
          kernel: kernelMap[kernel],
        });
        pixels.data.set(new Uint8ClampedArray(sharedArrayBuffer));
        context2D.putImageData(pixels, 0, 0);
        break;
      }

      default:
        context2D.putImageData(pixels, 0, 0);
        break;
    }

    // append image onto the canvas.

    let timeUsed = performance.now() - timeStart;
    filterTimeRecordsMap[filterOption].push(timeUsed);

    afterEachFrame(calcFPS(filterTimeRecordsMap[filterOption]));

    // continue.
    requestAnimationFrame(draw);
  };

  return {
    draw,
    // setUseWebWorker: (val: boolean) => (useWebWorker = val),
    setFilterOption: (val: FilterOption) => (filterOption = val),
    setKernel: (val: Kernel) => {
      kernel = val;
    },
  };
}
