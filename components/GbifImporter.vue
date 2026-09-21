<template>
  <div class="gbif-import-tool">
    <h3 class="text-lg font-semibold mb-4 text-gray-800">Import GBIF Occurrence Data</h3>

    <div class="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <p class="text-sm text-blue-800">
        <strong>How it works:</strong> Upload your GBIF CSV export file. We'll validate the coordinates,
        convert it to GeoJSON format, and prepare it for upload to your Earth Engine assets.
      </p>
    </div>

    <!-- File Upload Area -->
    <div
      class="border-2 border-dashed rounded-lg p-8 text-center transition-colors"
      :class="dragOver ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-gray-400'"
      @dragover.prevent="dragOver = true"
      @dragleave.prevent="dragOver = false"
      @drop.prevent="handleDrop"
    >
      <input
        ref="fileInput"
        type="file"
        accept=".csv"
        class="hidden"
        @change="handleFileSelect"
      />

      <div v-if="!selectedFile" class="space-y-3">
        <div class="text-4xl">📊</div>
        <p class="text-gray-600">Drag & drop your GBIF CSV file here</p>
        <p class="text-sm text-gray-500">or</p>
        <button
          @click="$refs.fileInput.click()"
          class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
        >
          Browse Files
        </button>
        <p class="text-xs text-gray-400">
          Download from <a href="https://www.gbif.org/occurrence/download" target="_blank" class="text-green-600 underline">GBIF.org</a>
        </p>
      </div>

      <div v-else class="space-y-3">
        <div class="text-4xl">✅</div>
        <p class="font-medium text-gray-800">{{ selectedFile.name }}</p>
        <p class="text-sm text-gray-500">{{ formatFileSize(selectedFile.size) }}</p>
        <div class="flex gap-2 justify-center">
          <button
            @click="resetFile"
            class="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
          >
            Choose Different File
          </button>
          <button
            @click="processFile"
            :disabled="processing"
            class="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:bg-gray-400 transition-colors"
          >
            {{ processing ? 'Processing...' : 'Process File' }}
          </button>
        </div>
      </div>
    </div>

    <!-- Processing Status -->
    <div v-if="processing || result" class="mt-6 space-y-4">
      <div v-if="processing" class="flex items-center gap-3 text-blue-600">
        <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
        </svg>
        <span>Processing your data...</span>
      </div>

      <!-- Results -->
      <div v-if="result" class="bg-white border border-gray-200 rounded-lg p-4 space-y-4">
        <div class="flex items-center gap-2 text-green-600">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <h4 class="font-semibold">Processing Complete!</h4>
        </div>

        <!-- Statistics -->
        <div class="grid grid-cols-3 gap-4">
          <div class="text-center p-3 bg-gray-50 rounded">
            <p class="text-2xl font-bold text-gray-800">{{ result.stats.totalRecords }}</p>
            <p class="text-xs text-gray-600">Total Records</p>
          </div>
          <div class="text-center p-3 bg-green-50 rounded">
            <p class="text-2xl font-bold text-green-600">{{ result.stats.validRecords }}</p>
            <p class="text-xs text-gray-600">Valid Points</p>
          </div>
          <div class="text-center p-3 bg-red-50 rounded">
            <p class="text-2xl font-bold text-red-600">{{ result.stats.skippedRecords }}</p>
            <p class="text-xs text-gray-600">Skipped</p>
          </div>
        </div>

        <!-- Asset Path -->
        <div class="p-3 bg-yellow-50 border border-yellow-200 rounded">
          <p class="text-sm font-medium text-yellow-800 mb-2">Next Step: Upload to Earth Engine</p>
          <code class="block text-xs bg-white p-2 rounded border border-yellow-200 break-all">
            {{ result.assetPath }}
          </code>
          <p class="text-xs text-yellow-700 mt-2">
            Copy this path and use it in the "Import Asset" feature, or manually upload the generated GeoJSON to your EE assets folder.
          </p>
        </div>

        <!-- Actions -->
        <div class="flex gap-2">
          <button
            @click="downloadGeoJSON"
            class="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm"
          >
            📥 Download GeoJSON
          </button>
          <button
            @click="loadToSession"
            class="flex-1 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors text-sm"
          >
            🗺️ Load to Current Session
          </button>
        </div>
      </div>

      <!-- Error -->
      <div v-if="error" class="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex items-start gap-2">
          <svg class="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
          <div>
            <p class="font-medium text-red-800">Error Processing File</p>
            <p class="text-sm text-red-700 mt-1">{{ error }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Help Section -->
    <div class="mt-6 pt-4 border-t border-gray-200">
      <details class="text-sm">
        <summary class="cursor-pointer text-gray-600 hover:text-gray-800 font-medium">
          ℹ️ What fields does GBIF export need?
        </summary>
        <div class="mt-2 text-gray-600 space-y-2">
          <p><strong>Required:</strong></p>
          <ul class="list-disc list-inside space-y-1 text-xs">
            <li><code>decimalLatitude</code> or <code>lat</code></li>
            <li><code>decimalLongitude</code> or <code>lon</code></li>
          </ul>
          <p class="mt-2"><strong>Recommended (optional):</strong></p>
          <ul class="list-disc list-inside space-y-1 text-xs">
            <li><code>species</code> or <code>scientificName</code></li>
            <li><code>eventDate</code> or <code>year</code></li>
            <li><code>gbifID</code></li>
            <li><code>coordinateUncertaintyInMeters</code></li>
          </ul>
        </div>
      </details>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const dragOver = ref(false)
const selectedFile = ref(null)
const processing = ref(false)
const result = ref(null)
const error = ref(null)

const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const handleDrop = (e) => {
  dragOver.value = false
  const files = e.dataTransfer.files
  if (files.length > 0 && files[0].name.endsWith('.csv')) {
    selectedFile.value = files[0]
  } else {
    error.value = 'Please upload a CSV file'
  }
}

const handleFileSelect = (e) => {
  const files = e.target.files
  if (files.length > 0) {
    selectedFile.value = files[0]
    error.value = null
  }
}

const resetFile = () => {
  selectedFile.value = null
  result.value = null
  error.value = null
}

const processFile = async () => {
  if (!selectedFile.value) return

  processing.value = true
  error.value = null
  result.value = null

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const response = await $fetch('/api/datasets/import-gbif', {
      method: 'POST',
      body: formData
    })

    result.value = response
  } catch (err) {
    error.value = err.message || 'Failed to process file'
    console.error('GBIF import error:', err)
  } finally {
    processing.value = false
  }
}

const downloadGeoJSON = () => {
  if (!result.value?.preview) return

  const blob = new Blob([JSON.stringify(result.value.preview, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `gbif_export_${result.value.datasetId}.geojson`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

const loadToSession = () => {
  // Emit event to parent component to add dataset to current session
  emit('dataset-loaded', {
    id: result.value.datasetId,
    name: `GBIF Import (${result.value.stats.validRecords} records)`,
    type: 'geojson',
    data: result.value.preview
  })
}

const emit = defineEmits(['dataset-loaded'])
</script>

<style scoped>
.gbif-import-tool {
  max-width: 600px;
  margin: 0 auto;
}

details summary::-webkit-details-marker {
  display: none;
}

details summary::before {
  content: '';
  display: inline-block;
  width: 16px;
  height: 16px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%236B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='9 18 15 12 9 6'%3E%3C/polyline%3E%3C/svg%3E");
  background-size: contain;
  background-repeat: no-repeat;
  vertical-align: middle;
  margin-right: 8px;
  transition: transform 0.2s;
}

details[open] summary::before {
  transform: rotate(90deg);
}
</style>
