import {WorkerPool} from './worker'
import {
  filter,
  return_pointer,
  take_pointer_by_value,
  filter_by_block,
} from 'rust-filter/rust_filter'
import {memory} from 'rust-filter/rust_filter_bg.wasm'

const workerPool = new WorkerPool(1)

export enum FilterOption {
  off = 'off',
  js = 'js',
  wasmGo = 'wasmGo',
  wasmRust = 'wasmRust',
}

const filterTimeRecordsMap: {[k: string]: number[]} = {
  [FilterOption.off]: [],
  [FilterOption.js]: [],
  [FilterOption.wasmGo]: [],
  [FilterOption.wasmRust]: [],
}

export enum Kernel {
  sharpen = 'sharpen',
  laplace = 'laplace',
  smoothing = 'smoothing',
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
}

function calcFPS(arr: number[]) {
  const n = 20
  if (arr.length > n) {
    arr.shift()
  }
  let averageTime =
    arr.reduce((pre, item) => {
      return pre + item
    }, 0) / arr.length
  return 1000 / averageTime
}

function getDenominator(kernel: number[][]) {
  let sum = 0
  for (let i = 0; i < kernel.length; i++) {
    const arr = kernel[i]
    for (let j = 0; j < arr.length; j++) {
      sum += Math.abs(kernel[i][j])
    }
  }
  return sum
}

function filterByJS(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  kernel: number[][]
) {
  // const newData = new Uint8ClampedArray(data)
  const h = kernel.length,
    w = h
  const half = Math.floor(h / 2)

  // picture iteration.
  for (let y = half; y < height - half; ++y) {
    for (let x = half; x < width - half; ++x) {
      const px = (y * width + x) * 4 // pixel index.
      let r = 0,
        g = 0,
        b = 0

      // core iteration.
      for (let cy = 0; cy < h; ++cy) {
        for (let cx = 0; cx < w; ++cx) {
          // dealing edge case.
          const cpx = ((y + (cy - half)) * width + (x + (cx - half))) * 4
          // if (px === 50216) debugger
          r += data[cpx + 0] * kernel[cy][cx]
          g += data[cpx + 1] * kernel[cy][cx]
          b += data[cpx + 2] * kernel[cy][cx]
          // a += data[cpx + 3] * kernel[cy][cx]
        }
      }
      data[px + 0] = r > 255 ? 255 : r < 0 ? 0 : r
      data[px + 1] = g > 255 ? 255 : g < 0 ? 0 : g
      data[px + 2] = b > 255 ? 255 : b < 0 ? 0 : b
      // data[px + 3] =
      //   a / denominator > 255 ? 255 : a / denominator < 0 ? 0 : a / denominator
    }
  }
  return data
}

type Pointer = number

function filterByGO(
  ptr: Pointer,
  width: number,
  height: number,
  kernel: number[][]
) {
  // @ts-ignore
  window.filterByGO(ptr, width, height, kernel)
}

async function filterByRust(
  ctxHidden: CanvasRenderingContext2D,
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  kernel: number[][]
) {
  filter(ctxHidden, ctx, width, height, new Float32Array([].concat(...kernel)))
}

function getSharedBuffer(imageData: Uint8ClampedArray) {
  const sharedArrayBuffer = new SharedArrayBuffer(imageData.buffer.byteLength)
  new Uint8ClampedArray(sharedArrayBuffer).set(imageData)
  return sharedArrayBuffer
}

function getPtr(imageData: Uint8ClampedArray) {
  const ptr = return_pointer()
  const uint8ClampedArray = new Uint8ClampedArray(memory.buffer)
  uint8ClampedArray.set(imageData)
  take_pointer_by_value(ptr)
  console.log(uint8ClampedArray[0])
  debugger
  return {ptr, uint8ClampedArray}
}

export function getDrawFn(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  canvasHidden: HTMLCanvasElement,
  afterEachFrame: (fps: number) => void,
  useWebWorker: boolean = false,
  filterOption: FilterOption = FilterOption.off,
  kernel: Kernel = Kernel.sharpen
) {
  const context2D = canvas.getContext('2d')!
  const context2DHidden = canvasHidden.getContext('2d')!
  const size = canvas.height * canvas.width * 4
  //@ts-ignore
  // For Go WASM
  // const {internalptr: ptr} = window.initShareMemory(size)
  // const mem = new Uint8ClampedArray(buffer, ptr, size)

  const draw = async () => {
    // record performance.
    const timeStart = performance.now()

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
    )

    context2DHidden.getImageData

    // get current video data.
    const pixels = context2DHidden.getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    )

    switch (filterOption) {
      case FilterOption.js: {
        if (useWebWorker) {
          const sharedArrayBuffer = getSharedBuffer(pixels.data)
          await workerPool.filter({
            sharedArrayBuffer,
            width: canvas.width,
            height: canvas.height,
            kernel: kernelMap[kernel],
          })
          pixels.data.set(new Uint8ClampedArray(sharedArrayBuffer))
        } else {
          pixels.data.set(
            filterByJS(
              pixels.data,
              canvas.width,
              canvas.height,
              kernelMap[kernel]
            )
          )
        }
        context2D.putImageData(pixels, 0, 0)
        break
      }
      // case FilterOption.wasmGo: {
      //   mem.set(pixels.data)
      //   filterByGO(ptr, canvas.width, canvas.height, kernelMap[kernel])
      //   // pixels.data.set(mem)
      //   // pixels.data.set(
      //   //   //@ts-ignore
      //   //   window.filterByGOCopy(
      //   //     pixels.data,
      //   //     canvas.width,
      //   //     canvas.height,
      //   //     kernelMap[kernel].flat()
      //   //   )
      //   // )
      //   context2D.putImageData(
      //     new ImageData(mem, canvas.width, canvas.height),
      //     0,
      //     0
      //   )

      //   break
      // }
      case FilterOption.wasmRust: {
        if (useWebWorker) {
          const {ptr, uint8ClampedArray} = getPtr(pixels.data)
          // filter_by_block(
          //   ptr,
          //   canvas.width,
          //   0,
          //   canvas.height,
          //   new Float32Array([].concat(...kernelMap[kernel]))
          // )
          // const sharedArrayBuffer = getSharedBuffer(pixels.data)
          debugger
          await workerPool.filter({
            useWasm: true,
            width: canvas.width,
            height: canvas.height,
            kernel: kernelMap[kernel],
          })
          pixels.data.set(uint8ClampedArray)

          context2D.putImageData(pixels, 0, 0)
        } else {
          filterByRust(
            context2DHidden,
            context2D,
            canvas.width,
            canvas.height,
            kernelMap[kernel]
          )
        }

        break
      }
      default:
        context2D.putImageData(pixels, 0, 0)
        break
    }

    // append image onto the canvas.

    let timeUsed = performance.now() - timeStart
    filterTimeRecordsMap[filterOption].push(timeUsed)

    afterEachFrame(calcFPS(filterTimeRecordsMap[filterOption]))

    // continue.
    requestAnimationFrame(draw)
  }

  return {
    draw,
    setUseWebWorker: (val: boolean) => (useWebWorker = val),
    setFilterOption: (val: FilterOption) => (filterOption = val),
    setKernel: (val: Kernel) => {
      kernel = val
    },
  }
}
