import { chromium } from 'playwright'
import fs from 'node:fs'
import path from 'node:path'

fs.mkdirSync('screenshots',{recursive:true})
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:980},deviceScaleFactor:1})
const base='http://127.0.0.1:5173'
const imageCache=new Map()

const photoQueries={
  'pottery-mug.png':'ceramic,mug,handmade',
  'pottery-vase.png':'ceramic,pottery,vase',
  'wood-ring-box.png':'wooden,jewelry,box',
  'wood-lamp.png':'wooden,table,lamp',
  'dyed-tote.png':'botanical,dye,tote,bag',
  'woven-basket.png':'wicker,picnic,basket',
  'embroidery-brooch.png':'embroidery,brooch,handmade',
  'embroidery-frame.png':'embroidery,flowers,hoop',
  'leather-notebook.png':'leather,journal,handmade',
  'leather-wallet.png':'leather,wallet,handmade',
  'silver-necklace.png':'silver,necklace,jewelry',
  'silver-ring.png':'silver,ring,jewelry',
  'tea-set.png':'porcelain,tea,set',
  'pottery-plate.png':'pottery,ceramic,plate',
  'wood-tray.png':'wooden,tray,handmade',
  'woven-coaster.png':'woven,coaster,handmade',
  'leather-keycase.png':'leather,keycase,handmade',
  'silver-earrings.png':'silver,earrings,jewelry'
}
const locks=Object.fromEntries(Object.keys(photoQueries).map((k,i)=>[k,201+i]))

await page.route('**/crafts-photo/*.png',async route=>{
  const name=new URL(route.request().url()).pathname.split('/').pop()
  try{
    if(!imageCache.has(name)){
      const query=photoQueries[name]||'handmade,craft'
      const r=await fetch('https://loremflickr.com/640/480/'+query+'?lock='+locks[name],{redirect:'follow',signal:AbortSignal.timeout(6000)})
      if(!r.ok)throw new Error('photo '+r.status)
      imageCache.set(name,Buffer.from(await r.arrayBuffer()))
    }
    await route.fulfill({status:200,contentType:'image/jpeg',body:imageCache.get(name)})
  }catch(e){
    const svg='public/crafts/'+name.replace('.png','.svg')
    if(fs.existsSync(svg))await route.fulfill({status:200,contentType:'image/svg+xml',body:fs.readFileSync(svg)})
    else await route.abort()
  }
})

async function wait(ms=550){await page.waitForTimeout(ms)}
async function shot(route,name){
  if(route!==null)await page.goto(base+route,{waitUntil:'networkidle'})
  await wait(650)
  await page.screenshot({path:'screenshots/'+name+'.png',fullPage:true})
  console.log('SHOT',name)
}
async function clear(){
  await page.goto(base+'/',{waitUntil:'domcontentloaded'})
  await page.evaluate(()=>localStorage.clear())
}
async function login(username){
  await page.goto(base+'/login',{waitUntil:'networkidle'})
  const inputs=page.locator('input')
  await inputs.nth(0).fill(username)
  await inputs.nth(1).fill('123456')
  await page.getByRole('button',{name:'登录',exact:true}).click()
  await wait(850)
}
async function api(method,url,body=null){
  return await page.evaluate(async({method,url,body})=>{
    const token=localStorage.getItem('craft-token')
    const headers={Authorization:'Bearer '+token}
    if(body!==null)headers['Content-Type']='application/json'
    const r=await fetch(url,{method,headers,body:body===null?undefined:JSON.stringify(body)})
    const text=await r.text();let data
    try{data=JSON.parse(text)}catch{data=text}
    return {status:r.status,data}
  },{method,url,body})
}

// ===== Public visitor =====
await clear()
await shot('/','01-public-home')
await shot('/works','02-public-works')
const publicWorks=await page.evaluate(()=>fetch('/api/works').then(r=>r.json()))
await shot('/works/'+publicWorks[0].id,'03-public-work-detail')
await shot('/login','04-login')
await shot('/register','05-register')

// ===== Customer =====
await clear();await login('user')
await page.goto(base+'/',{waitUntil:'networkidle'})
if(await page.locator('.home-like').count()){await page.locator('.home-like').first().click();await wait()}
await shot(null,'06-customer-home-like')

const works=(await api('GET','/api/works')).data
const firstId=works[0].id
const fav=(await api('GET','/api/favorites/work/'+firstId)).data
if(!fav.favorite)await api('POST','/api/favorites/work/'+firstId)
await shot('/works/'+firstId,'07-customer-work-detail-actions')
await shot('/favorites','08-customer-favorites')
await shot('/profile','09-customer-profile')

await page.goto(base+'/custom',{waitUntil:'networkidle'})
await page.getByPlaceholder('例如：毕业礼物木质首饰盒').fill('新居礼物·胡桃木桌面收纳架')
await page.getByPlaceholder('胡桃木、陶土、植鞣革...').fill('黑胡桃木')
await page.getByPlaceholder('尺寸、颜色、用途、刻字、参考风格等').fill('希望制作桌面收纳架，可放手机、钥匙和眼镜，边角圆润，正面刻名字缩写，整体偏简约自然风。')
await shot(null,'10-customer-custom-request')
await page.getByRole('button',{name:'发布需求，等待报价',exact:true}).click();await wait(900)
await shot(null,'11-customer-custom-published')

// Create a real quote/order for the newly published request so the payment screen is always testable,
// even when the earlier CI smoke flow has already completed the seeded pending order.
const myRequests=(await api('GET','/api/requests')).data
const paymentRequest=myRequests.find(r=>r.title==='新居礼物·胡桃木桌面收纳架')
if(paymentRequest){
  const creatorAuth=await page.evaluate(async()=>{
    const r=await fetch('/api/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:'creator',password:'123456'})})
    return await r.json()
  })
  const quoteCreated=await page.evaluate(async({token,requestId})=>{
    const r=await fetch('/api/quotes',{method:'POST',headers:{Authorization:'Bearer '+token,'Content-Type':'application/json'},body:JSON.stringify({requestId,price:328,leadDays:9,note:'黑胡桃木主体，分区收纳结构，边角手工圆磨，支持名字浅刻。'})})
    return await r.json()
  },{token:creatorAuth.token,requestId:paymentRequest.id})
  if(quoteCreated?.id)await api('POST','/api/orders/from-quote/'+quoteCreated.id)
}
// Keep another demand open so the creator demand hall and quote dialog have an actionable item.
await api('POST','/api/requests',{title:'纪念日手缝皮革卡套',category:'皮艺',budgetMin:180,budgetMax:320,preferredMaterial:'植鞣牛皮',dueDate:'2026-10-25',description:'希望制作一只深棕色手缝卡套，6个卡位，正面压印姓名首字母，适合作为纪念日礼物。'})

await shot('/orders','12-customer-orders-payment')
const pay=page.getByRole('button',{name:'模拟支付'}).first()
if(await pay.count()){await pay.click();await wait(900)}
await shot(null,'13-customer-orders-paid')

// ===== Merchant / creator =====
await clear();await login('creator')
await shot('/creator','14-creator-work-management')
await page.getByRole('button',{name:'+ 发布新作品',exact:true}).click();await wait(350)
const input=page.locator('input[type=file]')
if(await input.count()){
  await input.setInputFiles('public/crafts/pottery-vase.svg')
  await wait(750)
}
await shot(null,'15-creator-publish-upload')
const dialogCancel=page.getByRole('button',{name:'取消',exact:true})
if(await dialogCancel.count())await dialogCancel.last().click()
await wait(300)

await page.getByText('需求报价',{exact:true}).click();await wait(550)
await shot(null,'16-creator-demand-hall')
const quote=page.locator('button:has-text("给出报价"):not([disabled])').first()
if(await quote.count()){
  await quote.click();await wait(300)
  await shot(null,'17-creator-quote-dialog')
  const c=page.getByRole('button',{name:'取消',exact:true});if(await c.count())await c.last().click()
}
await page.getByText('制作与物流',{exact:true}).click();await wait(550)
await shot(null,'18-creator-production-orders')

const creatorOrders=(await api('GET','/api/orders')).data
const target=creatorOrders.find(o=>['待制作','制作中'].includes(o.status))
let drivenId=null
if(target){
  drivenId=target.id
  if(target.status==='待制作')await api('PUT','/api/orders/'+drivenId+'/progress',{progress:20,text:'已确认订单，开始准备材料'})
  await api('PUT','/api/orders/'+drivenId+'/progress',{progress:60,text:'主体制作完成，进入细节处理'})
  await api('PUT','/api/orders/'+drivenId+'/progress',{progress:100,text:'作品制作完成并完成质检，等待发货'})
  await api('POST','/api/logistics/order/'+drivenId+'/ship',{carrier:'顺丰速运（模拟）',trackingNo:''})
  await api('POST','/api/logistics/order/'+drivenId+'/advance')
  await page.goto(base+'/creator',{waitUntil:'networkidle'});await page.getByText('制作与物流',{exact:true}).click();await wait(650)
}
await shot(null,'19-creator-shipping-logistics')

// ===== Customer logistics + acceptance =====
await clear();await login('user')
await shot('/orders','20-customer-logistics-tracking')
if(drivenId){
  await clear();await login('creator');await api('POST','/api/logistics/order/'+drivenId+'/advance')
  await clear();await login('user');await shot('/orders','21-customer-acceptance')
  const accept=page.getByRole('button',{name:'确认验收'}).first()
  if(await accept.count()){await accept.click();await wait(800)}
  await shot(null,'22-customer-order-completed')
}

// ===== Admin =====
await clear();await login('admin')
await shot('/admin','23-admin-dashboard-echarts')
await page.getByText('用户管理',{exact:true}).click();await wait(500)
await shot(null,'24-admin-users')

const adminWorks=(await api('GET','/api/admin/works')).data
if(adminWorks.length)await api('PUT','/api/admin/works/'+adminWorks[0].id+'/status',{status:'PENDING'})
await page.goto(base+'/admin',{waitUntil:'networkidle'})
await page.getByText('作品审核',{exact:true}).click();await wait(550)
await shot(null,'25-admin-work-audit')
const pendingRow=page.locator('.el-table__row').filter({hasText:'待审核'}).first()
if(await pendingRow.count()){
  const approve=pendingRow.getByRole('button',{name:'通过'})
  if(await approve.count()){await approve.click();await wait(650)}
}
await shot(null,'26-admin-work-approved')

await page.getByText('评价管理',{exact:true}).click();await wait(500)
await shot(null,'27-admin-reviews')
await page.getByText('订单监管',{exact:true}).click();await wait(500)
await shot(null,'28-admin-orders')

await browser.close()
console.log('screenshots complete')
