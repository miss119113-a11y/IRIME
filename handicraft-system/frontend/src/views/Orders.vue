<script setup>
import {ref,onMounted} from 'vue'
import {completeOrder,getOrders} from '../api'
import {ElMessage} from 'element-plus'
const orders=ref([])
async function load(){const u=JSON.parse(localStorage.getItem('craft-user')||'null');if(!u)return;orders.value=(await getOrders({customerName:u.displayName})).data}
async function complete(id){await completeOrder(id);ElMessage.success('订单已确认完成');load()}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>我的定制订单</h1><p>从确认方案到成品交付，制作进度清晰可见。</p></div></section><section class="section"><div class="container">
<div class="panel order-card" v-for="o in orders" :key="o.id"><div><div style="display:flex;justify-content:space-between"><div><div class="work-title">{{o.workTitle}}</div><div style="font-size:12px;color:#81776d">创作者 · {{o.creatorName}}</div></div><span class="status">{{o.status}}</span></div><div style="margin:22px 0;color:#756d64;font-size:13px">订单金额 <b style="font-size:22px;color:#a65f3f;margin-left:8px">¥{{o.amount}}</b></div><div style="font-size:12px;color:#8b8177">创建时间：{{o.createdAt?.replace('T',' ').slice(0,16)}}　预计完成：{{o.expectedAt?.replace('T',' ').slice(0,16)}}</div></div>
<div class="progress-box"><b>制作进度 {{o.progress}}%</b><el-progress :percentage="o.progress" :stroke-width="9" color="#a65f3f" style="margin:14px 0"/><div style="font-size:12px;color:#776f66">{{o.progressText}}</div><el-button v-if="o.status==='待确认'" style="margin-top:14px;width:100%" @click="complete(o.id)">确认完成</el-button></div></div>
<div v-if="!orders.length" class="panel" style="text-align:center;color:#8d847b;padding:50px">还没有定制订单，先去发布一个需求吧。</div>
</div></section></template>
