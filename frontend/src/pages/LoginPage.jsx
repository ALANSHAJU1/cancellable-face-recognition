import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { authenticateUser } from "../services/api";
import "./LoginPage.css"; // ✅ NEW

function LoginPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [username, setUsername] = useState("");
  const [faceImage, setFaceImage] = useState(null);
  const [result, setResult] = useState("");

  const navigate = useNavigate();

  // -----------------------------
  // Start webcam
  // -----------------------------
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setResult("Camera started");
    } catch {
      setResult("Unable to access webcam");
    }
  };

  // -----------------------------
  // Stop webcam
  // -----------------------------
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      videoRef.current.srcObject = null;
    }
  };

  // -----------------------------
  // Capture face
  // -----------------------------
  const captureFace = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth) {
      setResult("Camera not ready. Adjust lighting.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) {
        setResult("Face capture failed");
        return;
      }

      setFaceImage(new File([blob], "auth_face.jpg", { type: "image/jpeg" }));
      stopCamera();
      setResult("Face image captured successfully");
    }, "image/jpeg");
  };

  // -----------------------------
  // Authenticate
  // -----------------------------
  const handleLogin = async () => {
    if (!username || !faceImage) {
      setResult("Username and face capture required");
      return;
    }

    setResult("Authenticating...");

    try {
      const formData = new FormData();
      formData.append("username", username);
      formData.append("face_image", faceImage);

      const data = await authenticateUser(formData);

      if (data.decision === "ACCEPT") {
        const safeUser = username.trim().toLowerCase().replace(/\s+/g, "_");
        navigate(`/dashboard/${safeUser}`);
      } else {
        setResult("Authentication failed");
      }
    } catch {
      setResult("Authentication error");
    }
  };

  return (
    <div className="login-bg">
      <div className="login-card">
        <h2>Sign In</h2>
        <p className="subtitle">Authenticate using live face capture</p>

        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <div className="camera-box">
          <video ref={videoRef} autoPlay />
        </div>

        <div className="btn-row">
          <button className="secondary" onClick={startCamera}>
            Start Camera
          </button>
          <button className="secondary" onClick={captureFace}>
            Capture Face
          </button>
        </div>

        <button className="primary" onClick={handleLogin}>
          Login
        </button>

        <p className="status">{result}</p>

        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>
    </div>
  );
}

export default LoginPage;
