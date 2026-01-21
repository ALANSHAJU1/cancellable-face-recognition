import { useRef, useState } from "react";
import { authenticateUser } from "../services/api";

function AuthenticatePage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [username, setUsername] = useState("");
  const [faceImage, setFaceImage] = useState(null);
  const [result, setResult] = useState("");

  // Start webcam
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

  // Stop webcam
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      videoRef.current.srcObject = null;
    }
  };

  // Capture live face
  const captureFace = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth) {
      setResult("Camera not ready. Please retry.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) {
        setResult("Face capture failed. Please retry.");
        return;
      }

      const file = new File([blob], "auth_face.jpg", {
        type: "image/jpeg",
      });

      setFaceImage(file);
      stopCamera();
      setResult("Face captured successfully");
    }, "image/jpeg");
  };

  // Authenticate
  const handleAuthenticate = async () => {
    if (!username || !faceImage) {
      setResult("Username and face capture are required");
      return;
    }

    setResult("Authenticating...");

    try {
      const formData = new FormData();
      formData.append("username", username);
      formData.append("face_image", faceImage);

      const response = await authenticateUser(formData);
      setResult(
        response.decision
          ? `Result: ${response.decision} (Score: ${response.score})`
          : "Authentication completed"
      );
    } catch {
      setResult("Authentication failed");
    }
  };

  return (
    <div>
      <h2>User Authentication</h2>

      <p>
        This interface verifies user identity by capturing a live facial image
        and comparing it with the stored protected biometric template.
      </p>

      {/* Username */}
      <div style={{ marginBottom: "10px" }}>
        <label>Username:</label><br />
        <input
          type="text"
          placeholder="Enter Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      </div>

      {/* Webcam Capture */}
      <div style={{ marginBottom: "10px" }}>
        <label>Live Face Capture:</label><br />
        <video ref={videoRef} autoPlay width="300" /><br />
        <button onClick={startCamera}>Start Camera</button>
        <button onClick={captureFace} style={{ marginLeft: "10px" }}>
          Capture Face
        </button>
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <button onClick={handleAuthenticate}>Authenticate</button>

      <p>{result}</p>
    </div>
  );
}

export default AuthenticatePage;
