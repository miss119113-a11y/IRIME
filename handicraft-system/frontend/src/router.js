import { createRouter, createWebHistory } from 'vue-router'
import Home from './views/Home.vue'
import Works from './views/Works.vue'
import Custom from './views/Custom.vue'
import Orders from './views/Orders.vue'
import Creator from './views/Creator.vue'
import Admin from './views/Admin.vue'
import Login from './views/Login.vue'

export default createRouter({
  history:createWebHistory(),
  routes:[
    {path:'/',component:Home},
    {path:'/works',component:Works},
    {path:'/custom',component:Custom},
    {path:'/orders',component:Orders},
    {path:'/creator',component:Creator},
    {path:'/admin',component:Admin},
    {path:'/login',component:Login},
  ]
})
