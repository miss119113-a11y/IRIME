import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as Icons from '@element-plus/icons-vue'
import App from './App.vue'
import router from './router'
import './style.css'
const app=createApp(App)
for(const [key,component] of Object.entries(Icons)) app.component(key,component)
app.use(ElementPlus).use(router).mount('#app')
