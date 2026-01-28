import { useNavigate } from "react-router-dom";
import "./CommonPages.css";

function HomePage() {
  const navigate = useNavigate();

  const handleExit = () => {
    // Browser-safe exit
    window.location.href = "about:blank";
  };

  return (
    <div className="page-bg">
      <div className="card">
        <h1 className="title">Cancellable Face Recognition System</h1>
        <p className="subtitle">
          Secure biometric authentication using cancellable templates
        </p>

        <button className="main-btn" onClick={() => navigate("/signup")}>
          Sign Up
        </button>

        <button className="main-btn" onClick={() => navigate("/login")}>
          Login
        </button>

        <button className="main-btn" onClick={handleExit}>
          Exit
        </button>
      </div>
    </div>
  );
}

export default HomePage;
