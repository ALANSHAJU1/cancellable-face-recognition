import axios from "axios";

const API_BASE = "http://localhost:5000/api";

export const enrollUser = async (formData) => {
  const res = await axios.post(`${API_BASE}/enroll`, formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
  return res.data;
};

export const authenticateUser = async (formData) => {
  const res = await axios.post(`${API_BASE}/authenticate`, formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
  return res.data;
};
