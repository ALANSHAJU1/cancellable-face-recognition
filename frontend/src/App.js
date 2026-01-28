
// import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

// import Dashboard from "./pages/Dashboard";
// import EnrollPage from "./pages/EnrollPage";
// import HomePage from "./pages/HomePage";
// import LoginPage from "./pages/LoginPage";
// import RegeneratePage from "./pages/RegeneratePage";

// function App() {
//   return (
//     <Router>
//       <Routes>
//         {/* Home Page */}
//         <Route path="/" element={<HomePage />} />

//         {/* Sign Up / Enrollment */}
//         <Route path="/signup" element={<EnrollPage />} />

//         {/* Login / Authentication */}
//         <Route path="/login" element={<LoginPage />} />

//         {/* Dashboard after successful authentication */}
//         <Route path="/dashboard/:username" element={<Dashboard />} />

//         {/* Regeneration after revocation */}
//         <Route path="/regenerate/:username" element={<RegeneratePage />} /> {/* ✅ ADDED */}
//       </Routes>
//     </Router>
//   );
// }

// export default App;


import { Route, BrowserRouter as Router, Routes } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import EnrollPage from "./pages/EnrollPage";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import RegeneratePage from "./pages/RegeneratePage";

function App() {
  return (
    <Router>
      <Routes>
        {/* Home */}
        <Route path="/" element={<HomePage />} />

        {/* Enrollment (Sign Up) */}
        <Route path="/signup" element={<EnrollPage />} />

        {/* Authentication (Login) */}
        <Route path="/login" element={<LoginPage />} />

        {/* Dashboard after successful authentication */}
        <Route path="/dashboard/:username" element={<Dashboard />} />

        {/* Regeneration after revocation (Design B) */}
        <Route path="/regenerate/:username" element={<RegeneratePage />} />

        {/* Fallback (safety) */}
        <Route path="*" element={<HomePage />} />
      </Routes>
    </Router>
  );
}

export default App;
