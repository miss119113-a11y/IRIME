<script setup>
import{ref,onMounted}from'vue';import{getFavorites,toggleFavorite}from'../api';import{ElMessage}from'element-plus'
const works=ref([])
async function load(){works.value=(await getFavorites()).data}
async function remove(w,e){e.stopPropagation();await toggleFavorite(w.id);works.value=works.value.filter(x=>x.id!==w.id);ElMessage.success('已移出收藏')}
onMounted(load)
</script><template><section class="page-hero"><div class="container"><h1>我的收藏</h1><p>保存想进一步了解或准备发起定制的作品。</p></div></section><section class="section"><div class="container"><div v-if="!works.length" class="empty-note">还没有收藏作品，去作品集看看吧。</div><div class="work-grid"><article class="work-card clickable" v-for="w in works" :key="w.id" @click="$router.push('/works/'+w.id)"><div class="work-image"><img :src="w.image" :alt="w.title"/><span class="tag">{{w.category}}</span><button class="icon-action card-fav-remove" @click="remove(w,$event)">★</button></div><div class="work-body"><div class="work-title">{{w.title}}</div><p class="work-summary">{{w.summary}}</p><div class="work-meta"><span>{{w.creatorName}}</span><span class="price">¥{{w.price}}</span></div></div></article></div></div></section></template>