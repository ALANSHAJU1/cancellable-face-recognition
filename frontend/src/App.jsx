import React from "react";
import EnrollPage from "./pages/EnrollPage";
import AuthenticatePage from "./pages/AuthenticatePage";

function App() {
  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>
      <h1>Cancellable Face Recognition System</h1>

      <EnrollPage />
      <hr />
      <AuthenticatePage />
    </div>
  );
}

export default App;
