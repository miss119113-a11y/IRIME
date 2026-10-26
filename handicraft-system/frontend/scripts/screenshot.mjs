import { chromium } from 'playwright'
import fs from 'node:fs'

fs.mkdirSync('screenshots',{recursive:true})
const browser=await chromium.launch({headless:true})
const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1})

async function shot(path,name,fullPage=true){
  await page.goto('http://127.0.0.1:5173'+path,{waitUntil:'networkidle'})
  await page.screenshot({path:'screenshots/'+name+'.png',fullPage})
}

await shot('/','01-home')
await shot('/works','02-works')
await shot('/custom','03-custom')
await shot('/creator','04-creator')
await shot('/admin','05-admin')

await page.goto('http://127.0.0.1:5173/login',{waitUntil:'networkidle'})
await page.locator('input').nth(0).fill('user')
await page.locator('input').nth(1).fill('123456')
await page.getByRole('button',{name:'登录',exact:true}).click()
await page.waitForURL('http://127.0.0.1:5173/')
await page.evaluate(()=>window.location.href='/orders')
await page.waitForLoadState('networkidle')
await page.screenshot({path:'screenshots/06-orders-login.png',fullPage:true})

await browser.close()
