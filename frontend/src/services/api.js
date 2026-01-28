import axios from "axios";

/*
  Backend base URL
  (DO NOT change – matches your working backend)
*/
const API_BASE = "http://localhost:5000/api";

/*
  Axios instance (clean & reusable)
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
    throw error.response?.data || { message: "Enrollment failed" };
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
    throw error.response?.data || { message: "Authentication failed" };
  }
};
