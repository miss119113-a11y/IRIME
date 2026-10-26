<script setup>
import {ref,reactive,onMounted} from 'vue'
import {getRequests,createQuote,getOrders,updateProgress} from '../api'
import {ElMessage} from 'element-plus'
const requests=ref([]),orders=ref([]),dialogs=ref({})
const form=reactive({price:0,leadDays:10,note:''})
async function load(){requests.value=(await getRequests()).data;orders.value=(await getOrders()).data}
async function quote(r){await createQuote({requestId:r.id,creatorName:'木屿手作',price:form.price,leadDays:form.leadDays,note:form.note});ElMessage.success('报价已提交');dialogs.value[r.id]=false;load()}
async function step(o,p,t){await updateProgress(o.id,{progress:p,text:t});ElMessage.success('制作进度已更新');load()}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>创作者工作台</h1><p>查看公开定制需求、提交方案报价，并持续更新制作进度。</p></div></section>
<section class="section"><div class="container"><div class="admin-grid"><div><div class="section-head"><div><h2 style="font-size:24px">定制需求大厅</h2><p>选择与你擅长方向匹配的需求。</p></div></div>
<div class="panel" v-for="r in requests" :key="r.id" style="margin-bottom:14px"><div style="display:flex;justify-content:space-between"><b>{{r.title}}</b><span class="status">{{r.status}}</span></div><p style="font-size:13px;line-height:1.7;color:#71685f">{{r.description}}</p><div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#8a8178"><span>{{r.category}} · ¥{{r.budgetMin}}-{{r.budgetMax}} · {{r.preferredMaterial}}</span><el-button size="small" @click="dialogs[r.id]=true">给出报价</el-button></div>
<el-dialog v-model="dialogs[r.id]" title="提交制作方案" width="430px"><el-form label-position="top"><el-form-item label="报价"><el-input-number v-model="form.price" :min="1" style="width:100%"/></el-form-item><el-form-item label="预计工期（天）"><el-input-number v-model="form.leadDays" :min="1" style="width:100%"/></el-form-item><el-form-item label="方案说明"><el-input v-model="form.note" type="textarea" :rows="4"/></el-form-item></el-form><template #footer><el-button @click="dialogs[r.id]=false">取消</el-button><el-button type="primary" @click="quote(r)">提交报价</el-button></template></el-dialog>
</div></div>
<div><div class="section-head"><div><h2 style="font-size:24px">制作中的订单</h2><p>更新真实制作节点。</p></div></div>
<div class="panel" v-for="o in orders.filter(x=>x.creatorName==='木屿手作'&&x.status!=='已完成')" :key="o.id" style="margin-bottom:14px"><b>{{o.workTitle}}</b><el-progress :percentage="o.progress" color="#a65f3f" style="margin:12px 0"/><div style="font-size:12px;color:#746c64;margin-bottom:12px">{{o.progressText}}</div><div style="display:flex;gap:6px"><el-button size="small" @click="step(o,55,'木盒主体已完成，正在进行结构细节处理')">主体完成</el-button><el-button size="small" @click="step(o,80,'表面处理完成，正在安装内衬并进行成品检查')">进入收尾</el-button><el-button size="small" type="primary" @click="step(o,100,'作品制作完成，等待用户确认')">制作完成</el-button></div></div>
</div></div></div></section></template>
