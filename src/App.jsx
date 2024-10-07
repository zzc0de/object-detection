import * as tf from "@tensorflow/tfjs";
import * as cocoModel from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";

function App() {
  const videoOption = useMemo(
    () => ({
      width: 480,
      height: 360,
      facingMode: "environment",
    }),
    []
  );

  const [dataset, setDataset] = useState(null);
  const [detections, setDetections] = useState([]);

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectIntervalRef = useRef(null);

  const loadModel = useCallback(async () => {
    try {
      const model = await cocoModel.load();
      setDataset(model);
    } catch (error) {
      console.log(error);
    }
  }, []);

  const drawBoundingBoxes = useCallback(
    (detections, canvasWidth, canvasHeight) => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      // Set canvas dimensions to match video
      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      detections.forEach((detection) => {
        const [x, y, width, height] = detection.bbox;
        const scaledX = canvasWidth - (x + width); // Flip X coordinate

        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 2;
        ctx.strokeRect(scaledX, y, width, height);

        ctx.fillStyle = "#00FFFF";
        ctx.font = "18px Arial";
        ctx.fillText(
          `${detection.class} - ${Math.round(detection.score * 100)}%`,
          scaledX,
          y > 10 ? y - 5 : 10
        );
      });
    },
    []
  );

  const detect = useCallback(async () => {
    if (dataset && webcamRef.current && canvasRef.current) {
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      // Set video width and height
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      const detections = await dataset.detect(video);
      setDetections(detections);
      drawBoundingBoxes(detections, videoWidth, videoHeight);
    }
  }, [dataset, drawBoundingBoxes]);

  useEffect(() => {
    tf.ready().then(() => {
      loadModel();
    });

    return () => {
      if (detectIntervalRef.current) {
        clearInterval(detectIntervalRef.current);
      }
    };
  }, [loadModel]);

  useEffect(() => {
    if (dataset) {
      detectIntervalRef.current = setInterval(detect, 100);
    }
    return () => {
      if (detectIntervalRef.current) {
        clearInterval(detectIntervalRef.current);
      }
    };
  }, [dataset, detect]);

  return (
    <>
      <nav className="w-full fixed bg-white flex justify-around items-center p-4">
        <h1 className="font-semibold text-xl">Halo</h1>
        <ul className="flex space-x-4">
          <li>
            <a href="#">About</a>
          </li>
          <li>
            <a href="/chat">ChatBot</a>
          </li>
        </ul>
      </nav>
      <div className="w-full h-screen bg-black text-white flex justify-center items-center">
        <div className="flex w-full justify-center items-center flex-col font-bold gap-y-6">
          <h1>Machine Learning by TensorFlow</h1>
          <div
            style={{
              position: "relative",
              width: videoOption.width,
              height: videoOption.height,
            }}
          >
            <Webcam
              ref={webcamRef}
              audio={false}
              videoConstraints={videoOption}
              style={{
                transform: "scaleX(-1)",
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "100%",
                height: "100%",
              }}
            />
          </div>
          <div>
            {detections.map((detection, index) => (
              <p key={index}>
                {detection.class} - {(detection.score * 100).toFixed(2)}%
              </p>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
