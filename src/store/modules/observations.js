import { supabase } from '@/plugins/supabase'

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
      const { data, error } = await supabase
        .from('observations')
        .select('id, inat_uuid, taxon_name, lat, lon, observed_on, place_guess, elevation, temp_avg, precip_total, wind_speed, ndvi, soil_moisture, cluster, prcp_d0, prcp_d1, prcp_d2, prcp_d3, prcp_d4, prcp_d5, prcp_d6')
        .eq('dataset_id', datasetId)
        .order('observed_on', { ascending: false })
      if (error) throw error
      commit('setObservations', { datasetId, observations: data })
      commit('setLoading', false)
    }
  },

  getters: {
    forDataset: state => datasetId => state.byDatasetId[datasetId] || []
  }
}
