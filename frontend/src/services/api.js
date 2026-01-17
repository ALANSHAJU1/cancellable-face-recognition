import axios from "axios";

const API_BASE = "http://localhost:5000/api";

export const enrollUser = async (userId) => {
  const res = await axios.post(`${API_BASE}/enroll`, {
    user_id: userId
  });
  return res.data;
};

export const authenticateUser = async (userId) => {
  const res = await axios.post(`${API_BASE}/authenticate`, {
    user_id: userId
  });
  return res.data;
};
