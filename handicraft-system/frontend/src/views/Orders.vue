<script setup>
import{ref,onMounted}from'vue';import{getOrders,completeOrder,cancelOrder,getPayment,payOrder,getLogistics}from'../api';import{ElMessage}from'element-plus'
const orders=ref([]),payment=ref({}),logistics=ref({})
async function load(){orders.value=(await getOrders()).data;for(const o of orders.value){try{payment.value[o.id]=(await getPayment(o.id)).data}catch{}try{logistics.value[o.id]=(await getLogistics(o.id)).data}catch{}}}
async function pay(o){await payOrder(o.id,{provider:'MOCK_WECHAT'});ElMessage.success('模拟支付成功');await load()}
async function accept(o){await completeOrder(o.id);ElMessage.success('验收完成');await load()}
async function cancel(o){await cancelOrder(o.id);ElMessage.success(o.status==='待支付'?'订单已取消':'取消申请已提交');await load()}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>我的定制订单</h1><p>支付、制作、发货、运输和验收全过程可追踪。</p></div></section><section class="section"><div class="container">
<div class="panel order-card-v2" v-for="o in orders" :key="o.id"><div class="order-main"><div class="order-title-row"><div><div class="work-title">{{o.workTitle}}</div><div class="subtle">创作者 · {{o.creatorName}}</div></div><span class="status">{{o.status}}</span></div>
<div class="order-facts"><span>订单金额 <b>¥{{o.amount}}</b></span><span>创建 {{o.createdAt?.replace('T',' ').slice(0,16)}}</span><span>预计 {{o.expectedAt?.replace('T',' ').slice(0,10)}}</span></div>
<el-steps :active="['待支付','已取消'].includes(o.status)?0:['待制作','制作中','取消申请中'].includes(o.status)?1:o.status==='待发货'?2:['已发货','运输中'].includes(o.status)?3:o.status==='待验收'?4:5" finish-status="success" simple class="order-steps"><el-step title="支付"/><el-step title="制作"/><el-step title="待发货"/><el-step title="物流"/><el-step title="验收"/><el-step title="完成"/></el-steps>
<div class="order-actions"><el-button v-if="o.status==='待支付'" type="primary" @click="pay(o)">模拟支付</el-button><el-button v-if="['待支付','待制作','制作中','待发货'].includes(o.status)" @click="cancel(o)">申请取消</el-button><el-button v-if="o.status==='待验收'" type="success" @click="accept(o)">确认验收</el-button></div>
</div><aside class="order-side"><div><b>当前进度 {{o.progress}}%</b><el-progress :percentage="o.progress" :stroke-width="8" color="#a65f3f" style="margin:12px 0"/><p>{{o.progressText}}</p></div>
<div v-if="payment[o.id]?.paymentNo" class="mini-section"><b>支付信息</b><p>{{payment[o.id].provider}} · {{payment[o.id].status}}</p><small>{{payment[o.id].paymentNo}}</small></div>
<div v-if="logistics[o.id]?.shipment" class="mini-section"><b>{{logistics[o.id].shipment.carrier}}</b><p>单号 {{logistics[o.id].shipment.trackingNo}}</p><div class="trace" v-for="t in logistics[o.id].traces" :key="t.id"><span></span><div><b>{{t.status}} · {{t.location}}</b><p>{{t.description}}</p></div></div></div>
</aside></div>
<div v-if="!orders.length" class="panel empty-note">还没有定制订单，先去发布一个需求吧。</div></div></section></template>
