import { api } from '@/plugins/api'

export default {
  namespaced: true,

  state: () => ({
    byDatasetId: {},
    loading: false
  }),

  mutations: {
    setObservations(state, { datasetId, observations }) {
      state.byDatasetId = { ...state.byDatasetId, [datasetId]: observations }
    },
    setLoading(state, v) { state.loading = v }
  },

  actions: {
    async fetchObservations({ commit }, datasetId) {
      commit('setLoading', true)
      try {
        const data = await api.listObservations(datasetId)
        commit('setObservations', { datasetId, observations: data })
      } finally {
        commit('setLoading', false)
      }
    }
  },

  getters: {
    forDataset: state => datasetId => state.byDatasetId[datasetId] || []
  }
}
