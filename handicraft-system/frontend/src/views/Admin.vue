<script setup>
import {ref,onMounted} from 'vue'
import {getDashboard,getOrders,getRequests,getWorks,getPayment,refundOrder} from '../api'
import {ElMessage,ElMessageBox} from 'element-plus'
const d=ref({}),orders=ref([]),requests=ref([]),works=ref([]),payment=ref({})
async function load(){
  d.value=(await getDashboard()).data
  orders.value=(await getOrders()).data
  requests.value=(await getRequests()).data
  works.value=(await getWorks()).data
  for(const o of orders.value){try{payment.value[o.id]=(await getPayment(o.id)).data}catch{}}
}
async function refund(o){
  try{
    const {value}=await ElMessageBox.prompt('请输入退款处理说明','模拟退款',{confirmButtonText:'确认退款',cancelButtonText:'取消',inputValue:'用户取消个性化定制订单'})
    await refundOrder(o.id,{reason:value})
    ElMessage.success('模拟退款完成，订单已关闭')
    await load()
  }catch(e){if(e!=='cancel'&&e!=='close'&&e?.message!=='cancel')console.error(e)}
}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>运营管理后台</h1><p>统一查看用户、作品、定制需求、支付退款与订单运行情况。</p></div></section>
<section class="section"><div class="container"><div class="dashboard-grid" style="grid-template-columns:repeat(6,1fr)"><div class="metric"><small>注册用户</small><b>{{d.users||0}}</b></div><div class="metric"><small>作品数量</small><b>{{d.works||0}}</b></div><div class="metric"><small>定制订单</small><b>{{d.orders||0}}</b></div><div class="metric"><small>已支付金额</small><b>¥{{d.paidAmount||0}}</b></div><div class="metric"><small>退款金额</small><b>¥{{d.refundAmount||0}}</b></div><div class="metric"><small>待退款</small><b>{{d.pendingRefund||0}}</b></div></div>
<div class="panel" v-if="orders.some(o=>o.status==='取消申请中')" style="margin-bottom:20px"><div class="section-head" style="margin-bottom:14px"><div><h2 style="font-size:22px">退款处理</h2><p>仅处理已经支付并由用户发起取消申请的定制订单。</p></div></div><el-table :data="orders.filter(o=>o.status==='取消申请中')"><el-table-column prop="workTitle" label="订单"/><el-table-column prop="customerName" label="用户" width="100"/><el-table-column prop="creatorName" label="创作者" width="110"/><el-table-column prop="amount" label="金额" width="100"><template #default="{row}">¥{{row.amount}}</template></el-table-column><el-table-column label="支付状态" width="110"><template #default="{row}">{{payment[row.id]?.status||'—'}}</template></el-table-column><el-table-column label="操作" width="120"><template #default="{row}"><el-button type="danger" size="small" @click="refund(row)">模拟退款</el-button></template></el-table-column></el-table></div>
<div class="admin-grid"><div class="panel"><h3 style="margin-top:0">最近定制需求</h3><el-table :data="requests.slice(0,8)"><el-table-column prop="title" label="需求"/><el-table-column prop="category" label="品类" width="90"/><el-table-column prop="userName" label="用户" width="100"/><el-table-column prop="status" label="状态" width="100"/></el-table></div>
<div class="panel"><h3 style="margin-top:0">订单与支付概览</h3><div v-for="o in orders.slice(0,8)" :key="o.id" class="soft-card" style="margin-bottom:10px"><div style="display:flex;justify-content:space-between"><b>{{o.workTitle}}</b><span class="status">{{o.status}}</span></div><div style="font-size:12px;color:#81776f;margin-top:7px">{{o.customerName}} → {{o.creatorName}} · ¥{{o.amount}}</div><div v-if="payment[o.id]?.paymentNo" style="font-size:11px;color:#9a8e84;margin-top:5px">{{payment[o.id].provider}} · {{payment[o.id].status}} · {{payment[o.id].paymentNo}}</div></div></div></div>
</div></section></template>
