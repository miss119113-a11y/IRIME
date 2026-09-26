<script setup>
import {ref,onMounted} from 'vue'
import {useRouter} from 'vue-router'
import {getWorks,getLikeStatus,toggleLike,toggleFavorite} from '../api'
import {ElMessage} from 'element-plus'
const router=useRouter(),works=ref([]),keyword=ref(''),category=ref(''),liked=ref({})
const user=JSON.parse(localStorage.getItem('craft-user')||'null')
const cats=['全部','陶艺','木作','编织','织染','刺绣','皮艺','银饰']
async function load(){works.value=(await getWorks({keyword:keyword.value,category:category.value})).data;if(user){for(const w of works.value){try{liked.value[w.id]=(await getLikeStatus(w.id)).data.liked}catch{}}}}
function pick(c){category.value=c==='全部'?'':c;load()}
async function like(w,e){e.stopPropagation();if(!user)return router.push('/login?redirect=/works');const {data}=await toggleLike(w.id);liked.value[w.id]=data.liked;w.likes=data.likes;ElMessage.success(data.liked?'已点赞':'已取消点赞')}
async function favorite(w,e){e.stopPropagation();if(!user)return router.push('/login?redirect=/works');const {data}=await toggleFavorite(w.id);ElMessage.success(data.favorite?'已加入收藏':'已取消收藏')}
onMounted(load)
</script>
<template><section class="page-hero"><div class="container"><h1>作品集</h1><p>从陶艺、木作、织染、刺绣、皮艺到银饰，发现可以真正定制的手工作品。</p></div></section>
<section class="section"><div class="container"><div class="toolbar"><el-input v-model="keyword" class="search" size="large" placeholder="搜索作品、创作者或关键词" clearable @keyup.enter="load"/><el-button size="large" @click="load">搜索</el-button></div>
<div class="category-row" style="margin-bottom:25px"><button v-for="c in cats" :key="c" class="category-chip" :class="{active:(c==='全部'&&!category)||category===c}" @click="pick(c)">{{c}}</button></div>
<div class="work-grid"><article class="work-card clickable" v-for="w in works" :key="w.id" @click="$router.push('/works/'+w.id)"><div class="work-image"><img :src="w.image" :alt="w.title"/><span class="tag">{{w.category}}</span><div class="card-actions"><button class="icon-action" :class="{active:liked[w.id]}" @click="like(w,$event)" title="喜欢">{{liked[w.id]?'♥':'♡'}} <small>{{w.likes}}</small></button><button class="icon-action" @click="favorite(w,$event)" title="收藏">☆</button></div></div><div class="work-body"><div class="work-title">{{w.title}}</div><p class="work-summary">{{w.summary}}</p><div class="work-meta"><span>约 {{w.leadDays}} 天</span><span class="price">¥{{w.price}}</span></div><div class="creator">创作者 · {{w.creatorName}}</div></div></article></div>
</div></section></template>