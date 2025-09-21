// src/services/api.ts
import axios from "axios";

// Set backend base URL (change if deployed)
const API = axios.create({
  baseURL: "http://localhost:8000", // backend FastAPI URL
  headers: {
    "Content-Type": "application/json",
  },
});

// -------- AUTH --------
export const login = (data: { username: string; password: string }) =>
  API.post("/login", data);

export const signup = (data: { username: string; password: string }) =>
  API.post("/signup", data);

// -------- HISTORY --------
export const getHistory = () => API.get("/");

export const addHistory = (data: { input: string; result: string }) =>
  API.post("/add", data);

// -------- PREDICT --------
export const predictWithPkl = (data: { input: string }) =>
  API.post("/pkl-best", data);

export const predictWithOnnx = (data: { input: string }) =>
  API.post("/onnx", data);
