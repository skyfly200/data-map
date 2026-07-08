import { api } from '@/plugins/api'

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
      commit('setError', null)
      try {
        const data = await api.listDatasets()
        commit('setDatasets', data)
      } catch (e) {
        commit('setError', e.message)
      } finally {
        commit('setLoading', false)
      }
    },

    async createDataset({ commit }, { name, description, config }) {
      // The server creates the dataset and kicks off the pipeline automatically.
      const dataset = await api.createDataset({ name, description, config })
      commit('addDataset', dataset)
      return dataset
    },

    async deleteDataset({ commit }, id) {
      await api.deleteDataset(id)
      commit('removeDataset', id)
    }
  },

  getters: {
    completedDatasets: state => state.datasets.filter(d => d.status === 'complete'),
    byId: state => id => state.datasets.find(d => d.id === id)
  }
}
