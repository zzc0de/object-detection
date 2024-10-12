import * as tf from "@tensorflow/tfjs";
import * as cocoModel from "@tensorflow-models/coco-ssd";
import Webcam from "react-webcam";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";

function App() {
  // setup camera
  const videoOption = useMemo(
    () => ({
      width: 480,
      height: 360,
      facingMode: "environment",
    }),
    []
  );

  const [dataset, setDataset] = useState(null); //state untuk menyimpan model
  const [detections, setDetections] = useState([]); //state untuk menyimpan deteksi

  const webcamRef = useRef(null); //ref untuk webcam
  const canvasRef = useRef(null); //ref untuk canvas
  const detectIntervalRef = useRef(null); //ref untuk interval

  // load model
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

      // Set dimensi canvas ke ukuran video
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

  // use effect untuk load data ketika halaman pertama kali dibuka
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
      <nav className="fixed flex items-center justify-around w-full p-4 bg-white">
        <h1 className="text-xl font-semibold">
          <a href="/">Object Detection</a>
        </h1>
        <ul className="flex space-x-4">
          <li>
            <a href="/">About</a>
          </li>
          <li>
            <a href="/chatbot">ChatBot</a>
          </li>
        </ul>
      </nav>
      <div className="flex items-center justify-center w-full h-screen text-white bg-black">
        <div className="flex flex-col items-center justify-center w-full font-bold gap-y-6">
          <h1>Machine Learning Object Detection</h1>
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
              className="sm:transform-none"
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
