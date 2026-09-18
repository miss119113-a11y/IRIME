<script setup>
import {ref,onMounted} from 'vue'
import {getDashboard,getOrders,getRequests,getWorks} from '../api'
const d=ref({}),orders=ref([]),requests=ref([]),works=ref([])
onMounted(async()=>{d.value=(await getDashboard()).data;orders.value=(await getOrders()).data;requests.value=(await getRequests()).data;works.value=(await getWorks()).data})
</script>
<template><section class="page-hero"><div class="container"><h1>运营管理后台</h1><p>统一查看用户、作品、定制需求与订单运行情况。</p></div></section>
<section class="section"><div class="container"><div class="dashboard-grid"><div class="metric"><small>注册用户</small><b>{{d.users||0}}</b></div><div class="metric"><small>作品数量</small><b>{{d.works||0}}</b></div><div class="metric"><small>定制需求</small><b>{{d.requests||0}}</b></div><div class="metric"><small>定制订单</small><b>{{d.orders||0}}</b></div><div class="metric"><small>订单金额</small><b>¥{{d.revenue||0}}</b></div></div>
<div class="admin-grid"><div class="panel"><h3 style="margin-top:0">最近定制需求</h3><el-table :data="requests.slice(0,8)"><el-table-column prop="title" label="需求"/><el-table-column prop="category" label="品类" width="90"/><el-table-column prop="userName" label="用户" width="100"/><el-table-column prop="status" label="状态" width="100"/></el-table></div>
<div class="panel"><h3 style="margin-top:0">订单概览</h3><div v-for="o in orders.slice(0,6)" :key="o.id" class="soft-card" style="margin-bottom:10px"><div style="display:flex;justify-content:space-between"><b>{{o.workTitle}}</b><span class="status">{{o.status}}</span></div><div style="font-size:12px;color:#81776f;margin-top:7px">{{o.customerName}} → {{o.creatorName}} · ¥{{o.amount}}</div></div></div></div>
</div></section></template>
