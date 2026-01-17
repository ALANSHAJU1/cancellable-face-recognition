import { useState } from "react";
import { enrollUser } from "../services/api";

function EnrollPage() {
  const [userId, setUserId] = useState("");
  const [status, setStatus] = useState("");

  const handleEnroll = async () => {
    setStatus("Enrolling user...");
    try {
      const result = await enrollUser(userId);
      setStatus(result.message || "Enrollment completed");
    } catch {
      setStatus("Enrollment failed");
    }
  };

  return (
    <div>
      <h2>User Enrollment</h2>

      <p>
        This interface initiates the enrollment process by sending the user ID
        to the backend, which triggers the biometric enrollment pipeline.
      </p>

      <input
        type="text"
        placeholder="Enter Username"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />

      <button onClick={handleEnroll} style={{ marginLeft: "10px" }}>
        Enroll
      </button>

      <p>{status}</p>
    </div>
  );
}

export default EnrollPage;
