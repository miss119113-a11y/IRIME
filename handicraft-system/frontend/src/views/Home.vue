<script setup>
import { ref,onMounted } from 'vue'
import { getFeatured } from '../api'
const works=ref([])
onMounted(async()=>{try{works.value=(await getFeatured()).data}catch(e){console.error(e)}})
</script>
<template>
<section class="hero"><div class="container hero-grid">
  <div>
    <div class="eyebrow">HANDMADE · CUSTOM · STORY</div>
    <h1>把你的想法，<br/>交给<em>手艺人</em>实现</h1>
    <p>发现有温度的原创手工艺，连接创作者与喜欢独特作品的你。选一件喜欢的作品，或从一个想法开始，让它成为只属于你的实物。</p>
    <div class="hero-actions"><button class="primary-btn" @click="$router.push('/works')">探索作品</button><button class="outline-btn" @click="$router.push('/custom')">发布定制需求</button></div>
  </div>
  <div class="hero-art"><img src="https://images.unsplash.com/photo-1455390582262-044cdead277a?auto=format&fit=crop&w=1200&q=85"/><div class="floating-note"><b>从草图到成品</b>需求 · 报价 · 制作 · 交付，全程可追踪</div></div>
</div></section>

<section class="section"><div class="container">
  <div class="section-head"><div><h2>本周精选手作</h2><p>从陶艺、木作、织染到皮艺，每一件都保留手工痕迹。</p></div><RouterLink to="/works" class="text-link">查看全部 →</RouterLink></div>
  <div class="work-grid">
    <article class="work-card" v-for="w in works.slice(0,6)" :key="w.id">
      <div class="work-image"><img :src="w.image"/><span class="tag">{{w.category}}</span><span class="heart">♡</span></div>
      <div class="work-body"><div class="work-title">{{w.title}}</div><div class="work-meta"><span>约 {{w.leadDays}} 天制作</span><span class="price">¥{{w.price}}</span></div><div class="creator">创作者 · {{w.creatorName}}　 · {{w.likes}} 人喜欢</div></div>
    </article>
  </div>
</div></section>

<section class="feature-band"><div class="container">
  <div class="section-head" style="margin-bottom:36px"><div><h2 style="color:white">个性定制，不止是下单</h2><p style="color:#c7bfb7">把模糊想法变成清晰方案，再把方案变成作品。</p></div></div>
  <div class="steps"><div class="step"><span class="step-no">STEP 01</span><h3>发布需求</h3><p>填写品类、预算、材质、尺寸和期望完成时间。</p></div><div class="step"><span class="step-no">STEP 02</span><h3>创作者报价</h3><p>不同创作者给出制作方案、周期和价格。</p></div><div class="step"><span class="step-no">STEP 03</span><h3>确认制作</h3><p>选择满意方案生成订单，制作过程持续更新。</p></div><div class="step"><span class="step-no">STEP 04</span><h3>完成交付</h3><p>查看成品、确认完成并留下真实评价。</p></div></div>
</div></section>
</template>
