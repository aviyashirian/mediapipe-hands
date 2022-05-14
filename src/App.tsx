// Import dependencies
import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
import "./App.css";
import * as mpHands from '@mediapipe/hands'
import * as drawingUtils from '@mediapipe/drawing_utils'
import * as controls from '@mediapipe/control_utils'

function App() {
  // const webcamRef = useRef<Webcam>(null);
  // const canvasRef = useRef<HTMLCanvasElement>(null);

  // Main function
  const runApp = async () => {    
    const net = await tf.loadLayersModel('http://localhost:3000/saved_model/model.json')

    detectHands(net)
  };

  
  const detectHands = (net: tf.LayersModel) => {
        
    // Our input frames will come from here.
    const videoElement =
      document.getElementsByClassName('input_video')[0] as HTMLVideoElement;
    const canvasElement =
      document.getElementsByClassName('output_canvas')[0] as HTMLCanvasElement;
    const controlsElement =
      document.getElementsByClassName('control-panel')[0] as HTMLDivElement;
    const canvasCtx = canvasElement.getContext('2d')!;

    const config = {locateFile: (file: string) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}/${file}`;
    }};

    // We'll add this to our control panel later, but we'll save it here so we can
    // call tick() each time the graph runs.
    const fpsControl = new controls.FPS();

    const onResults = (results: mpHands.Results): void => {
      // Update the frame rate.
      fpsControl.tick();

      // Draw the overlays.
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height);
      if (results.multiHandLandmarks && results.multiHandedness) {
        for (let index = 0; index < results.multiHandLandmarks.length; index++) {
          const classification = results.multiHandedness[index];
          const isRightHand = classification.label === 'Right';
          const landmarks = results.multiHandLandmarks[index];
          drawingUtils.drawConnectors(
              canvasCtx, landmarks, mpHands.HAND_CONNECTIONS,
              {color: isRightHand ? '#00FF00' : '#FF0000'});
          drawingUtils.drawLandmarks(canvasCtx, landmarks, {
            color: isRightHand ? '#00FF00' : '#FF0000',
            fillColor: isRightHand ? '#FF0000' : '#00FF00',
            radius: (data: drawingUtils.Data) => {
              return drawingUtils.lerp(data.from!.z!, -0.15, .1, 10, 1);
            }
          });
        }
      }
  
      canvasCtx.restore();

      if (results.multiHandWorldLandmarks) {
        // We only get to call updateLandmarks once, so we need to cook the data to
        // fit. The landmarks just merge, but the connections need to be offset.
        const landmarks = results.multiHandWorldLandmarks.reduce(
            (prev, current) => [...prev, ...current], []).flatMap(landmark => [landmark.x, landmark.y])
        
        const colors = [];
        let connections: mpHands.LandmarkConnectionArray = [];
        for (let loop = 0; loop < results.multiHandWorldLandmarks.length; ++loop) {
          const offset = loop * mpHands.HAND_CONNECTIONS.length;
          const offsetConnections =
              mpHands.HAND_CONNECTIONS.map(
                  (connection) =>
                      [connection[0] + offset, connection[1] + offset]) as
              mpHands.LandmarkConnectionArray;
          connections = connections.concat(offsetConnections);
          const classification = results.multiHandedness[loop];
          colors.push({
            list: offsetConnections.map((unused, i) => i + offset),
            color: classification.label
          });
        }

        if (landmarks.length === 42) {

          // predict hand signature
          const landmarksTensor = tf.tensor(landmarks).expandDims().expandDims()
          const arr = (net.predict(landmarksTensor) as tf.Tensor[])
          const pred = arr[0].dataSync() as Float32Array
          const prediction = getMaxPrediction(pred)
          const THRESHOLD = 0.7;
          if (prediction.probability > THRESHOLD) {
            console.log(prediction.class, prediction.probability)

            writeTextOnCanvas(canvasCtx, {text: prediction.class.toString(), x: 50, y: 0})
            writeTextOnCanvas(canvasCtx, {text: (prediction.probability * 100).toFixed(2).toString() + '%', x: 50, y: 50}, { color: 'green' })
          }
        }
      // grid.updateLandmarks(landmarks, connections, colors);
      } else {
      // grid.updateLandmarks([]);
      }
    }

    const hands = new mpHands.Hands(config);
    hands.onResults(onResults);

    // Present a control panel through which the user can manipulate the solution
    // options.
    new controls
    .ControlPanel(controlsElement, {
      selfieMode: true,
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    .add([
      // new controls.StaticText({title: 'MediaPipe Hands'}),
      fpsControl,
      // new controls.Toggle({title: 'Selfie Mode', field: 'selfieMode'}),
      new controls.SourcePicker({
        onFrame:
            async (input: controls.InputImage, size: controls.Rectangle) => {
              const aspect = size.height / size.width;
              let width: number, height: number;
              if (window.innerWidth > window.innerHeight) {
                height = window.innerHeight;
                width = height / aspect;
              } else {
                width = window.innerWidth;
                height = width * aspect;
              }
              canvasElement.width = width;
              canvasElement.height = height;
              await hands.send({image: input});
            },
      })
    ])
    .on(newOptions => {
      const options = newOptions as mpHands.Options;
      videoElement.classList.toggle('selfie', options.selfieMode);
      hands.setOptions(options);
    });
  }

  const getMaxPrediction = (preds: Float32Array): {class: number, probability: number} => {
    const indexOfMaxValue = preds.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);
    const probability = preds[indexOfMaxValue]
  
    return {class: indexOfMaxValue, probability}
  }

   // write a text
  const writeTextOnCanvas = (
    ctx: CanvasRenderingContext2D,
    info: {text: string, x: number, y: number},
    style: {color: string} = { color: 'black'}
  ): void => {
    const { text, x, y } = info;
    const color = style.color;
    const fontSize = 60;
    const fontFamily = 'Arial';
    const textAlign = 'left';
    const textBaseline = 'top'

    ctx.beginPath();
    ctx.font = fontSize + 'px ' + fontFamily;
    ctx.textAlign = textAlign;
    ctx.textBaseline = textBaseline;
    ctx.fillStyle = color;
    ctx.fillText(text, x, y);
    ctx.stroke();
  }

  useEffect(() => { runApp() });

  return (
    <div>
      <div className="container">
        <video className="input_video"></video>
        <canvas className="output_canvas" width="1280px" height="720px"></canvas>

      </div>
      <div className="control-panel">
      </div>
    </div>
  );
}

export default App;
