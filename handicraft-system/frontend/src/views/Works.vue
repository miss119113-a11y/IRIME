<script setup>
import {ref,onMounted,watch} from 'vue'
import {getWorks} from '../api'
const works=ref([]),keyword=ref(''),category=ref('')
const cats=['全部','陶艺','木作','织染','刺绣','皮艺']
async function load(){works.value=(await getWorks({keyword:keyword.value,category:category.value})).data}
function pick(c){category.value=c==='全部'?'':c;load()}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>作品集</h1><p>寻找一种材质、一位创作者，或者一个能触发灵感的细节。</p></div></section>
<section class="section"><div class="container"><div class="toolbar"><el-input v-model="keyword" class="search" size="large" placeholder="搜索作品、创作者或关键词" clearable @keyup.enter="load"/><el-button size="large" @click="load">搜索</el-button></div>
<div class="category-row" style="margin-bottom:25px"><button v-for="c in cats" :key="c" class="category-chip" :class="{active:(c==='全部'&&!category)||category===c}" @click="pick(c)">{{c}}</button></div>
<div class="work-grid"><article class="work-card" v-for="w in works" :key="w.id"><div class="work-image"><img :src="w.image"/><span class="tag">{{w.category}}</span><span class="heart">♡</span></div><div class="work-body"><div class="work-title">{{w.title}}</div><p style="font-size:12px;color:#80766d;line-height:1.7;height:42px;overflow:hidden">{{w.summary}}</p><div class="work-meta"><span>约 {{w.leadDays}} 天</span><span class="price">¥{{w.price}}</span></div><div class="creator">创作者 · {{w.creatorName}}</div></div></article></div>
</div></section></template>
