<script setup>
import {reactive,ref,onMounted} from 'vue'
import {ElMessage} from 'element-plus'
import {createRequest,getRequests,getQuotes,createOrderFromQuote} from '../api'
const user=()=>JSON.parse(localStorage.getItem('craft-user')||'null')
const list=ref([]),quoteMap=ref({})
const form=reactive({title:'',category:'木作',budgetMin:200,budgetMax:500,preferredMaterial:'',dueDate:'',description:''})
async function load(){const u=user();if(!u)return;list.value=(await getRequests({userName:u.displayName})).data;for(const r of list.value)quoteMap.value[r.id]=(await getQuotes(r.id)).data}
async function submit(){const u=user();if(!u){ElMessage.warning('请先登录');return}await createRequest({...form,userName:u.displayName});ElMessage.success('定制需求已发布');Object.assign(form,{title:'',category:'木作',budgetMin:200,budgetMax:500,preferredMaterial:'',dueDate:'',description:''});load()}
async function choose(q){await createOrderFromQuote(q.id);ElMessage.success('已选择报价并生成定制订单');load()}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>个性定制</h1><p>描述你的想法，让创作者给出制作方案与报价。</p></div></section>
<section class="section"><div class="container" style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
<div class="panel"><h3 style="margin-top:0;font-family:'Noto Serif SC'">发布新的定制需求</h3><div class="form-grid">
<el-form-item label="需求名称"><el-input v-model="form.title" placeholder="例如：毕业礼物木质首饰盒"/></el-form-item>
<el-form-item label="作品品类"><el-select v-model="form.category" style="width:100%"><el-option v-for="c in ['木作','陶艺','织染','刺绣','皮艺','其他']" :key="c" :label="c" :value="c"/></el-select></el-form-item>
<el-form-item label="预算下限"><el-input-number v-model="form.budgetMin" :min="0" style="width:100%"/></el-form-item>
<el-form-item label="预算上限"><el-input-number v-model="form.budgetMax" :min="0" style="width:100%"/></el-form-item>
<el-form-item label="偏好材质"><el-input v-model="form.preferredMaterial" placeholder="胡桃木、陶土、植鞣革..."/></el-form-item>
<el-form-item label="期望日期"><el-date-picker v-model="form.dueDate" type="date" value-format="YYYY-MM-DD" style="width:100%"/></el-form-item>
<el-form-item label="详细描述" class="form-full"><el-input v-model="form.description" type="textarea" :rows="5" placeholder="尺寸、颜色、用途、刻字、参考风格等"/></el-form-item>
</div><button class="primary-btn" style="width:100%" @click="submit">发布需求，等待报价</button></div>
<div><div class="section-head" style="margin-bottom:15px"><div><h2 style="font-size:24px">我的定制需求</h2><p>报价会展示在对应需求下方。</p></div></div>
<div v-for="r in list" :key="r.id" class="panel" style="margin-bottom:15px"><div style="display:flex;justify-content:space-between;gap:10px"><b>{{r.title}}</b><span class="status">{{r.status}}</span></div><p style="font-size:13px;color:#746b62;line-height:1.7">{{r.description}}</p><div style="font-size:12px;color:#8a8178">预算 ¥{{r.budgetMin}} - ¥{{r.budgetMax}} · {{r.category}} · {{r.preferredMaterial}}</div>
<div v-if="quoteMap[r.id]?.length" style="margin-top:14px"><div class="soft-card" v-for="q in quoteMap[r.id]" :key="q.id" style="margin-top:8px"><div style="display:flex;justify-content:space-between"><b>{{q.creatorName}}</b><strong style="color:#a65f3f">¥{{q.price}}</strong></div><p style="font-size:12px;color:#756e66">{{q.note}}</p><div style="display:flex;align-items:center;justify-content:space-between;font-size:12px"><span>制作周期 {{q.leadDays}} 天</span><el-button v-if="r.status!=='已转订单'" size="small" @click="choose(q)">选择此方案</el-button></div></div></div>
<div v-else style="margin-top:12px;color:#9a928a;font-size:12px">暂未收到报价</div></div>
</div></div></section></template>
