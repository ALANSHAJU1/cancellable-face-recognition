import { useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { enrollUser } from "../services/api";
import "./EnrollPage.css"; // ✅ SAME CSS (important)

function RegeneratePage() {
  const { username } = useParams();
  const navigate = useNavigate();

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [faceImage, setFaceImage] = useState(null);
  const [facePreview, setFacePreview] = useState(null);
  const [coverImage, setCoverImage] = useState(null);
  const [status, setStatus] = useState("");
  const [cameraOn, setCameraOn] = useState(false);

  // -----------------------------
  // Start camera
  // -----------------------------
  const startCamera = async () => {
    if (cameraOn) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setCameraOn(true);
      setStatus("Camera started. Position your face properly.");
    } catch {
      setStatus("Unable to access webcam.");
    }
  };

  // -----------------------------
  // Stop camera
  // -----------------------------
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
      videoRef.current.srcObject = null;
      setCameraOn(false);
    }
  };

  // -----------------------------
  // Capture face
  // -----------------------------
  const captureFace = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video.videoWidth) {
      setStatus("Face not detected. Improve lighting.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(blob => {
      if (!blob) {
        setStatus("Face capture failed.");
        return;
      }

      const file = new File([blob], "face.jpg", { type: "image/jpeg" });
      setFaceImage(file);
      setFacePreview(URL.createObjectURL(blob));
      setStatus("Face image captured successfully");
      stopCamera();
    }, "image/jpeg");
  };

  // -----------------------------
  // Regenerate Template
  // -----------------------------
  const handleRegenerate = async () => {
    if (!faceImage || !coverImage) {
      setStatus("Face image and cover image are required.");
      return;
    }

    setStatus("Regenerating biometric template...");

    try {
      const formData = new FormData();
      formData.append("username", username);
      formData.append("face_image", faceImage);
      formData.append("cover_image", coverImage);

      await enrollUser(formData);

      alert("Biometric template regenerated successfully.");
      navigate(`/dashboard/${username}`);
    } catch {
      setStatus("Regeneration failed. Please retry.");
    }
  };

  return (
    <div className="signup-bg">
      <div className="signup-card">
        <h2>Regenerate Template</h2>
        <p className="subtitle">
          Your previous biometric template was revoked.
          Capture a new face image and upload a new cover image
          to regenerate a fresh cancellable template.
        </p>

        {/* Username (LOCKED) */}
        <input
          type="text"
          value={username}
          disabled
        />

        {/* CAMERA */}
        {!facePreview && (
          <>
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
          </>
        )}

        {/* FACE PREVIEW */}
        {facePreview && (
          <>
            <img className="preview" src={facePreview} alt="Face Preview" />
            <p className="success">Face image captured successfully</p>
          </>
        )}

        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* COVER IMAGE */}
        <input
          type="file"
          accept="image/*"
          onChange={e => setCoverImage(e.target.files[0])}
        />

        <button
          className="primary"
          onClick={handleRegenerate}
          disabled={!faceImage || !coverImage}
        >
          Regenerate
        </button>

        <p className="status">{status}</p>
      </div>
    </div>
  );
}

export default RegeneratePage;
