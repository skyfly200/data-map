import { createStore } from 'vuex'
import datasets from './modules/datasets'
import observations from './modules/observations'

export default createStore({
  modules: { datasets, observations }
})
