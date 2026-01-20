import { useRef, useState } from "react";
import { enrollUser } from "../services/api";

function EnrollPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [username, setUsername] = useState("");
  const [faceImage, setFaceImage] = useState(null);
  const [coverImage, setCoverImage] = useState(null);
  const [status, setStatus] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);

  // Start webcam
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setIsCameraOn(true);
      setStatus("Camera started");
    } catch (err) {
      console.error(err);
      setStatus("Unable to access webcam");
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      videoRef.current.srcObject = null;
      setIsCameraOn(false);
    }
  };

  // Capture face image
  const captureFace = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!isCameraOn || !video.videoWidth) {
      setStatus("Camera not ready. Please retry.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setStatus("Face capture failed. Please retry.");
          stopCamera();
          return;
        }

        const file = new File([blob], "face.jpg", {
          type: "image/jpeg"
        });

        setFaceImage(file);
        setStatus("Face captured successfully. Camera stopped.");
        stopCamera();
      },
      "image/jpeg",
      0.95
    );
  };

  const handleEnroll = async () => {
    if (!username || !faceImage || !coverImage) {
      setStatus(
        "Please provide username, capture face, and upload cover image."
      );
      return;
    }

    setStatus("Enrolling user...");

    try {
      const formData = new FormData();
      formData.append("username", username);
      formData.append("face_image", faceImage);
      formData.append("cover_image", coverImage);

      const result = await enrollUser(formData);
      setStatus(result.message || "Enrollment completed successfully");
    } catch (err) {
      console.error(err);
      setStatus("Enrollment failed");
    }
  };

  return (
    <div>
      <h2>User Enrollment</h2>

      <p>
        This interface performs user enrollment by capturing the facial image
        through a webcam and embedding the biometric information into a
        user-selected cover image.
      </p>

      <div style={{ marginBottom: "10px" }}>
        <label>Username:</label><br />
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
      </div>

      <div style={{ marginBottom: "10px" }}>
        <label>Face Capture (Webcam):</label><br />
        <video ref={videoRef} autoPlay width="300" /><br />
        <button onClick={startCamera}>Start Camera</button>
        <button
          onClick={captureFace}
          disabled={!isCameraOn}
          style={{ marginLeft: "10px" }}
        >
          Capture Face
        </button>
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>

      <div style={{ marginBottom: "10px" }}>
        <label>Cover Image:</label><br />
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setCoverImage(e.target.files[0])}
        />
      </div>

      <button onClick={handleEnroll}>Enroll</button>
      <p>{status}</p>
    </div>
  );
}

export default EnrollPage;
