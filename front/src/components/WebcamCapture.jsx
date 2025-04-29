import React, { useRef, useEffect, useState } from "react";

const WebcamCapture = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState("");

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" }, // usa la cámara frontal o integrada
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accediendo a la cámara:", err);
      }
    };

    startCamera();
  }, []);

  const captureAndSend = () => {
    const context = canvasRef.current.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, 224, 224);

    canvasRef.current.toBlob((blob) => {
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      fetch("https://tu-backend.up.railway.app/predict", {
        method: "POST",
        body: formData,
      })
        .then((res) => res.json())
        .then((data) => {
          setPrediction(data.prediction);
        })
        .catch((err) => console.error("Error:", err));
    }, "image/jpeg");
  };

  return (
    <div style={{ textAlign: "center", marginTop: "2rem" }}>
      <video ref={videoRef} autoPlay playsInline width="224" height="224" />
      <canvas ref={canvasRef} width="224" height="224" style={{ display: "none" }} />
      <br />
      <button onClick={captureAndSend} style={{ marginTop: "1rem" }}>
        Analizar expresión
      </button>
      {prediction && <h3>Emoción detectada: {prediction}</h3>}
    </div>
  );
};

export default WebcamCapture;
