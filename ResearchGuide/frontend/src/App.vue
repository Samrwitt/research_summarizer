<template>
  <main class="workspace">
    <aside class="rail">
      <div class="brand-block">
        <p class="eyebrow">CorpusMind</p>
        <h1>Research command center</h1>
      </div>

      <label class="dropzone" :class="{ ready: file }">
        <UploadCloud :size="24" />
        <span>{{ fileName || 'Select a research file' }}</span>
        <small>PDF, text, markdown, or CSV</small>
        <input type="file" accept=".pdf,.txt,.md,.csv" @change="onFile" />
      </label>

      <button class="primary-action" :disabled="!file || loading" @click="analyze">
        <LoaderCircle v-if="loading" class="spin" :size="18" />
        <Sparkles v-else :size="18" />
        <span>{{ loading ? 'Building intelligence layer' : 'Analyze document' }}</span>
      </button>

      <nav class="nav-stack" aria-label="Workspace views">
        <button
          v-for="item in views"
          :key="item.id"
          :class="{ active: activeView === item.id }"
          :disabled="!analysis"
          @click="activeView = item.id"
        >
          <component :is="item.icon" :size="17" />
          <span>{{ item.label }}</span>
        </button>
      </nav>

      <section v-if="analysis" class="pipeline">
        <div>
          <span class="signal good"></span>
          <strong>{{ analysis.rag.embedding_model }}</strong>
          <small>BERT retrieval model</small>
        </div>
        <div>
          <span class="signal"></span>
          <strong>{{ analysis.rag.retrieval }}</strong>
          <small>ranking strategy</small>
        </div>
      </section>
      <section v-else-if="serviceStatus" class="pipeline">
        <div>
          <span class="signal" :class="{ good: serviceStatus.ai_service?.status === 'ok' }"></span>
          <strong>{{ serviceStatus.ai_service?.service || 'AI service' }}</strong>
          <small>{{ serviceStatus.ai_service?.status || 'unavailable' }}</small>
        </div>
        <div>
          <span class="signal good"></span>
          <strong>{{ serviceStatus.capabilities?.length || 0 }} capabilities</strong>
          <small>Django orchestration layer</small>
        </div>
      </section>
    </aside>

    <section class="surface">
      <header class="topbar">
        <div>
          <p class="eyebrow">Evidence Workspace</p>
          <h2>{{ analysis?.title || 'No document indexed yet' }}</h2>
        </div>
        <div class="status-pill" :class="{ online: analysis }">
          <Database :size="16" />
          <span>{{ analysis ? 'Indexed' : 'Waiting' }}</span>
        </div>
      </header>

      <div v-if="error" class="error-banner">
        <AlertTriangle :size="18" />
        <span>{{ error }}</span>
      </div>

      <section v-if="!analysis" class="empty-state">
        <div class="empty-copy">
          <p class="eyebrow">Start</p>
          <h2>Upload a paper to generate summaries, citations, table evidence, and image OCR signals.</h2>
        </div>
        <div class="preview-shell">
          <div class="preview-bar"></div>
          <div class="preview-grid">
            <span></span><span></span><span></span><span></span>
            <span></span><span></span><span></span><span></span>
          </div>
        </div>
      </section>

      <template v-else>
        <section class="metric-grid">
          <article v-for="metric in metrics" :key="metric.label" class="metric-tile">
            <component :is="metric.icon" :size="20" />
            <strong>{{ metric.value }}</strong>
            <span>{{ metric.label }}</span>
          </article>
        </section>

        <section v-show="activeView === 'overview'" class="layout-grid">
          <article class="panel summary-panel">
            <div class="panel-heading">
              <h3>Executive Summary</h3>
              <FileText :size="19" />
            </div>
            <p>{{ analysis.summary }}</p>
            <div class="keywords">
              <span v-for="keyword in analysis.insights.keywords" :key="keyword">{{ keyword }}</span>
            </div>
          </article>

          <article class="panel findings-panel">
            <div class="panel-heading">
              <h3>Key Findings</h3>
              <ListChecks :size="19" />
            </div>
            <ul>
              <li v-for="bullet in analysis.bullets" :key="bullet">{{ bullet }}</li>
            </ul>
          </article>
        </section>

        <section v-show="activeView === 'ask'" class="qa-layout">
          <div class="question-bar">
            <Search :size="19" />
            <input v-model="question" placeholder="Ask about methods, results, tables, figures, or limitations" @keydown.enter="ask" />
            <button :disabled="!question || asking" @click="ask">
              <LoaderCircle v-if="asking" class="spin" :size="18" />
              <ArrowRight v-else :size="18" />
            </button>
          </div>

          <div class="prompt-row">
            <button v-for="prompt in prompts" :key="prompt" @click="question = prompt">{{ prompt }}</button>
          </div>

          <article v-if="answer" class="panel answer-panel">
            <div class="panel-heading">
              <h3>Grounded Answer</h3>
              <Quote :size="19" />
            </div>
            <p class="answer-text">{{ answer.answer }}</p>
          </article>
        </section>

        <section v-show="activeView === 'evidence'" class="evidence-layout">
          <article class="panel">
            <div class="panel-heading">
              <h3>Citations</h3>
              <Braces :size="19" />
            </div>
            <div v-if="answer?.citations?.length" class="citation-list">
              <button
                v-for="citation in answer.citations"
                :key="citation.id"
                :class="{ selected: selectedCitation === citation.id }"
                @click="selectedCitation = citation.id"
              >
                <span>{{ citation.source_type }}</span>
                <strong>p.{{ citation.page || 'n/a' }}</strong>
                <small>{{ citation.score.toFixed(3) }}</small>
              </button>
            </div>
            <p v-else class="muted">Ask a question to populate ranked evidence.</p>
          </article>

          <article class="panel evidence-text">
            <div class="panel-heading">
              <h3>Retrieved Context</h3>
              <Rows3 :size="19" />
            </div>
            <div v-if="selectedContext">
              <p>{{ selectedContext.text }}</p>
              <code>{{ selectedContext.id }}</code>
            </div>
            <p v-else class="muted">No context selected.</p>
          </article>
        </section>
      </template>
    </section>
  </main>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import {
  AlertTriangle,
  ArrowRight,
  Braces,
  Database,
  FileText,
  Image,
  Layers3,
  LayoutDashboard,
  ListChecks,
  LoaderCircle,
  Quote,
  Rows3,
  Search,
  Sparkles,
  Table2,
  UploadCloud
} from 'lucide-vue-next'
import { askDocument, getIntelligenceStatus, uploadDocument } from './services/api'

const file = ref(null)
const fileName = ref('')
const loading = ref(false)
const asking = ref(false)
const analysis = ref(null)
const question = ref('')
const answer = ref(null)
const activeView = ref('overview')
const selectedCitation = ref('')
const serviceStatus = ref(null)
const error = ref('')

const views = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'ask', label: 'Ask', icon: Search },
  { id: 'evidence', label: 'Evidence', icon: Braces }
]

const prompts = [
  'What is the main contribution?',
  'Which results are supported by tables?',
  'What limitations should I mention?'
]

const metrics = computed(() => {
  if (!analysis.value) return []
  const modalities = analysis.value.rag.modalities
  return [
    { label: 'BERT chunks', value: analysis.value.rag.chunk_count, icon: Layers3 },
    { label: 'tables parsed', value: modalities.tables, icon: Table2 },
    { label: 'OCR blocks', value: modalities.image_ocr_blocks, icon: Image },
    { label: 'keywords', value: analysis.value.insights.keywords?.length || 0, icon: Braces }
  ]
})

const selectedContext = computed(() => {
  if (!answer.value?.contexts?.length) return null
  return answer.value.contexts.find((context) => context.id === selectedCitation.value) || answer.value.contexts[0]
})

watch(answer, (value) => {
  selectedCitation.value = value?.citations?.[0]?.id || ''
})

getIntelligenceStatus()
  .then((data) => {
    serviceStatus.value = data
  })
  .catch(() => {
    serviceStatus.value = { backend: 'unavailable', ai_service: { status: 'unavailable' }, capabilities: [] }
  })

function onFile(event) {
  file.value = event.target.files[0]
  fileName.value = file.value?.name || ''
}

async function analyze() {
  loading.value = true
  answer.value = null
  error.value = ''
  activeView.value = 'overview'
  try {
    const data = await uploadDocument(file.value)
    analysis.value = data.analysis
  } catch (failure) {
    error.value = failure?.response?.data?.detail || 'Document analysis failed.'
  } finally {
    loading.value = false
  }
}

async function ask() {
  if (!question.value.trim()) return
  asking.value = true
  error.value = ''
  try {
    answer.value = await askDocument(analysis.value.document_id, question.value)
    activeView.value = 'evidence'
  } catch (failure) {
    error.value = failure?.response?.data?.detail || 'Question answering failed.'
  } finally {
    asking.value = false
  }
}
</script>
