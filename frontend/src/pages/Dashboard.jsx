import axios from "axios";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import "./CommonPages.css";

function Dashboard() {
  const { username } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  // -----------------------------
  // Revoke Template (Design B)
  // -----------------------------
  const handleRevoke = async () => {
    if (loading) return;

    const confirmRevoke = window.confirm(
      "Are you sure you want to revoke your biometric template?\n\n" +
      "This will permanently invalidate your current biometric data.\n" +
      "You will NOT be able to authenticate unless you regenerate a new template."
    );

    if (!confirmRevoke) return;

    setLoading(true);

    try {
      const response = await axios.post(
        "http://localhost:5000/api/revoke",
        {
          // ✅ CRITICAL FIX: normalize username
          username: username.trim().toLowerCase()
        }
      );

      if (response.data && response.data.success) {
        const regenerateNow = window.confirm(
          "Template revoked successfully.\n\n" +
          "All cryptographic keys have been deleted.\n\n" +
          "Do you want to regenerate a new biometric template now?"
        );

        if (regenerateNow) {
          navigate(`/regenerate/${username}`);
        } else {
          navigate("/");
        }
      } else {
        alert(response.data.message || "Revocation failed.");
      }
    } catch (error) {
      console.error("Revocation error:", error);
      alert("Revocation failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-bg">
      <div className="card">
        <h1 className="title">Welcome</h1>
        <h2 className="username">{username}</h2>

        <p className="subtitle">
          You have been successfully authenticated
        </p>

        <button
          className="main-btn"
          onClick={() => navigate("/")}
          disabled={loading}
        >
          Logout
        </button>

        <button
          className="main-btn danger"
          onClick={handleRevoke}
          disabled={loading}
        >
          {loading ? "Revoking..." : "Revoke Template"}
        </button>
      </div>
    </div>
  );
}

export default Dashboard;
