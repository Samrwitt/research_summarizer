import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api'
})

export async function uploadDocument(file) {
  const body = new FormData()
  body.append('file', file)
  const { data } = await api.post('/documents/upload/', body)
  return data
}

export async function getIntelligenceStatus() {
  const { data } = await api.get('/intelligence/status/')
  return data
}

export async function askDocument(documentId, question) {
  const { data } = await api.post(`/documents/${documentId}/ask/`, { question, top_k: 8 })
  return data
}
