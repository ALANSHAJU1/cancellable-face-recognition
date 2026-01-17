import React, { useState } from "react";
import { authenticateUser } from "../services/api";

function AuthenticatePage() {
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState("");

  const handleAuthenticate = async () => {
    setResult("Authenticating...");
    try {
      const response = await authenticateUser(userId);
      setResult(response.decision || response.message);
    } catch {
      setResult("Authentication failed");
    }
  };

  return (
    <div>
      <h2>User Authentication</h2>

      <p>
        This interface performs identity verification by comparing the live
        biometric input with the stored protected template.
      </p>

      <input
        type="text"
        placeholder="Enter User ID"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />

      <button onClick={handleAuthenticate} style={{ marginLeft: "10px" }}>
        Authenticate
      </button>

      <p>{result}</p>
    </div>
  );
}

export default AuthenticatePage;
