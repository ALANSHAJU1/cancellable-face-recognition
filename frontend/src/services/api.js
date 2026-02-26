import axios from "axios";

/*
  Backend base URL
*/
const API_BASE = "http://localhost:5000/api";

/*
  Axios instance
*/
const api = axios.create({
  baseURL: API_BASE,
});

/*
  -----------------------
  USER ENROLLMENT
  -----------------------
*/
export const enrollUser = async (formData) => {
  try {
    const res = await api.post("/enroll", formData);
    return res.data;
  } catch (error) {
    // 🔥 IMPORTANT: Throw full error object
    throw error;
  }
};

/*
  -----------------------
  USER AUTHENTICATION
  -----------------------
*/
export const authenticateUser = async (formData) => {
  try {
    const res = await api.post("/authenticate", formData);
    return res.data;
  } catch (error) {
    throw error;
  }
};
