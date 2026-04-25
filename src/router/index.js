import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import DatasetsView from '@/views/DatasetsView.vue'
import DatasetView from '@/views/DatasetView.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: HomeView },
    { path: '/datasets', component: DatasetsView },
    { path: '/datasets/:id', component: DatasetView }
  ]
})
