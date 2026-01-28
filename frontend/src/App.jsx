import { BrowserRouter, Route, Routes } from "react-router-dom";

import AuthenticatePage from "./pages/AuthenticatePage";
import Dashboard from "./pages/Dashboard";
import EnrollPage from "./pages/EnrollPage";
import HomePage from "./pages/HomePage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Home */}
        <Route path="/" element={<HomePage />} />

        {/* Sign Up / Enrollment */}
        <Route path="/signup" element={<EnrollPage />} />

        {/* Login / Authentication */}
        <Route path="/login" element={<AuthenticatePage />} />

        {/* Dashboard */}
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
