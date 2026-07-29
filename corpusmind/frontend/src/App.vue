<template>
  <main class="workspace">
    <aside class="sidebar">
      <div>
        <p class="eyebrow">CorpusMind</p>
        <h1>Multimodal research intelligence</h1>
      </div>
      <label class="upload">
        <UploadCloud :size="22" />
        <span>{{ fileName || 'Upload PDF or text' }}</span>
        <input type="file" accept=".pdf,.txt,.md,.csv" @change="onFile" />
      </label>
      <button :disabled="!file || loading" @click="analyze">
        <Sparkles :size="18" />
        <span>{{ loading ? 'Analyzing' : 'Analyze' }}</span>
      </button>
      <div v-if="analysis" class="stats">
        <div><strong>{{ analysis.rag.chunk_count }}</strong><span>indexed chunks</span></div>
        <div><strong>{{ analysis.rag.modalities.tables }}</strong><span>tables</span></div>
        <div><strong>{{ analysis.rag.modalities.image_ocr_blocks }}</strong><span>OCR blocks</span></div>
      </div>
    </aside>

    <section class="content">
      <header class="topbar">
        <div>
          <p class="eyebrow">Evidence workspace</p>
          <h2>{{ analysis?.title || 'Ready for a document' }}</h2>
        </div>
        <Database :size="24" />
      </header>

      <section v-if="analysis" class="grid">
        <article class="panel summary">
          <h3>Executive Summary</h3>
          <p>{{ analysis.summary }}</p>
          <div class="keywords">
            <span v-for="keyword in analysis.insights.keywords" :key="keyword">{{ keyword }}</span>
          </div>
        </article>

        <article class="panel">
          <h3>Key Findings</h3>
          <ul>
            <li v-for="bullet in analysis.bullets" :key="bullet">{{ bullet }}</li>
          </ul>
        </article>
      </section>

      <section class="ask">
        <div class="question">
          <input v-model="question" :disabled="!analysis" placeholder="Ask a question about the paper, tables, or images" />
          <button :disabled="!analysis || !question || asking" @click="ask">
            <Search :size="18" />
          </button>
        </div>
        <article v-if="answer" class="panel answer">
          <h3>Grounded Answer</h3>
          <p class="answer-text">{{ answer.answer }}</p>
          <div class="citations">
            <span v-for="citation in answer.citations" :key="citation.id">
              {{ citation.source_type }} p.{{ citation.page || 'n/a' }} score {{ citation.score.toFixed(2) }}
            </span>
          </div>
        </article>
      </section>
    </section>
  </main>
</template>

<script setup>
import { ref } from 'vue'
import { Database, Search, Sparkles, UploadCloud } from 'lucide-vue-next'
import { askDocument, uploadDocument } from './services/api'

const file = ref(null)
const fileName = ref('')
const loading = ref(false)
const asking = ref(false)
const analysis = ref(null)
const question = ref('')
const answer = ref(null)

function onFile(event) {
  file.value = event.target.files[0]
  fileName.value = file.value?.name || ''
}

async function analyze() {
  loading.value = true
  answer.value = null
  try {
    const data = await uploadDocument(file.value)
    analysis.value = data.analysis
  } finally {
    loading.value = false
  }
}

async function ask() {
  asking.value = true
  try {
    answer.value = await askDocument(analysis.value.document_id, question.value)
  } finally {
    asking.value = false
  }
}
</script>
