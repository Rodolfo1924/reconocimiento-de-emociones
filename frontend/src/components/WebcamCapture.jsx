import React, { useRef, useEffect, useState } from "react";
import './styles.css';

const WebcamCapture = ({ onCapture }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let stream = null;

    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setError("No se pudo acceder a la cámara. Por favor, verifica los permisos.");
        console.error("Error accediendo a la cámara:", err);
      }
    };

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const captureAndSend = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    setIsLoading(true);
    setError(null);
    setPrediction("");

    const context = canvasRef.current.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, 224, 224);

    canvasRef.current.toBlob(async (blob) => {
      try {
        const data = await onCapture(blob);
        if (data && data.prediction) {
          setPrediction(data.prediction);
        } else {
          setError("No se recibió una predicción válida del backend.");
        }
      } catch (err) {
        setError("Error al analizar la expresión. Intenta de nuevo.");
        console.error("Error al enviar la imagen:", err);
      } finally {
        setIsLoading(false);
      }
    }, "image/jpeg");
  };

  return (
    <div className="section" style={{ textAlign: "center" }}>
      <video ref={videoRef} autoPlay playsInline width="224" height="224" />
      <canvas ref={canvasRef} width="224" height="224" style={{ display: "none" }} />
      <br />
      <button onClick={captureAndSend} className="button" disabled={isLoading}>
        {isLoading ? "Analizando..." : "Analizar expresión"}
      </button>
      {prediction && <h3 style={{ color: "#2E3192", marginTop: "1rem" }}>Emoción detectada: {prediction}</h3>}
      {error && <p style={{ color: "#FF6F61", marginTop: "1rem" }}>{error}</p>}
    </div>
  );
};

export default WebcamCapture;