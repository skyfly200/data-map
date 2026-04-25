import { supabase } from '@/plugins/supabase'

export default {
  namespaced: true,

  state: () => ({
    datasets: [],
    loading: false,
    error: null
  }),

  mutations: {
    setDatasets(state, datasets) { state.datasets = datasets },
    addDataset(state, dataset) { state.datasets.unshift(dataset) },
    updateDataset(state, updated) {
      const i = state.datasets.findIndex(d => d.id === updated.id)
      if (i !== -1) state.datasets.splice(i, 1, updated)
    },
    removeDataset(state, id) {
      state.datasets = state.datasets.filter(d => d.id !== id)
    },
    setLoading(state, v) { state.loading = v },
    setError(state, e) { state.error = e }
  },

  actions: {
    async fetchDatasets({ commit }) {
      commit('setLoading', true)
      const { data, error } = await supabase
        .from('datasets')
        .select('*')
        .order('created_at', { ascending: false })
      if (error) { commit('setError', error.message); return }
      commit('setDatasets', data)
      commit('setLoading', false)
    },

    async createDataset({ commit }, { name, description, config }) {
      const { data, error } = await supabase
        .from('datasets')
        .insert({ name, description, config, status: 'pending', stage: 'idle' })
        .select()
        .single()
      if (error) throw error

      commit('addDataset', data)

      // Kick off the pipeline — fire and forget (202 background function)
      await fetch('/.netlify/functions/pipeline-inat-background', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: data.id })
      })

      return data
    },

    async deleteDataset({ commit }, id) {
      const { error } = await supabase.from('datasets').delete().eq('id', id)
      if (error) throw error
      commit('removeDataset', id)
    }
  },

  getters: {
    completedDatasets: state => state.datasets.filter(d => d.status === 'complete'),
    byId: state => id => state.datasets.find(d => d.id === id)
  }
}
