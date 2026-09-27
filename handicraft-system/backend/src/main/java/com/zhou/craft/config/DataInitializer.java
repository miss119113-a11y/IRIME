package com.zhou.craft.config;
import com.zhou.craft.model.*;import com.zhou.craft.repo.*;import org.springframework.boot.CommandLineRunner;import org.springframework.context.annotation.*;import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;import java.time.LocalDate;
@Configuration public class DataInitializer{
@Bean CommandLineRunner init(UserRepository users,CraftWorkRepository works,CustomRequestRepository requests,QuoteRepository quotes,CraftOrderRepository orders,PaymentRecordRepository payments,WorkReviewRepository reviews){
return args->{var encoder=new BCryptPasswordEncoder();
ensureUser(users,encoder,"admin","123456","admin@craft.local","系统管理员","ADMIN","https://api.dicebear.com/9.x/notionists/svg?seed=admin");
ensureUser(users,encoder,"creator","123456","creator@craft.local","木屿手作","CREATOR","https://api.dicebear.com/9.x/notionists/svg?seed=craft");
ensureUser(users,encoder,"user","123456","user@craft.local","周欣悦","USER","https://api.dicebear.com/9.x/notionists/svg?seed=zhou");
var creator=users.findByUsername("creator").orElseThrow();
seedWorks(works,creator.getId());
if(requests.count()==0){var user=users.findByUsername("user").orElseThrow();
var r1=requests.save(new CustomRequest(user.getId(),user.getDisplayName(),"毕业礼物木质首饰盒","木作",280,450,"希望制作一个可放项链、耳饰和戒指的胡桃木首饰盒，外盖刻名字缩写 ZXY。","胡桃木",LocalDate.now().plusDays(22)));var q1=quotes.save(new Quote(r1.getId(),creator.getId(),creator.getDisplayName(),368,12,"胡桃木主体+米色绒布内衬，支持名字浅刻。"));r1.setStatus("已转订单");requests.save(r1);orders.save(new CraftOrder(r1.getId(),q1.getId(),user.getId(),creator.getId(),r1.getTitle(),user.getDisplayName(),creator.getDisplayName(),q1.getPrice(),q1.getLeadDays()));
var r2=requests.save(new CustomRequest(user.getId(),user.getDisplayName(),"暖色系陶瓷咖啡杯","陶艺",120,220,"喜欢暖棕、米白色，杯身希望有不规则手作纹理，容量约350ml。","陶土",LocalDate.now().plusDays(15)));var q2=quotes.save(new Quote(r2.getId(),creator.getId(),creator.getDisplayName(),188,8,"暖白釉面搭配手作纹理，杯底刻名字缩写。"));r2.setStatus("已转订单");requests.save(r2);var o2=new CraftOrder(r2.getId(),q2.getId(),user.getId(),creator.getId(),r2.getTitle(),user.getDisplayName(),creator.getDisplayName(),q2.getPrice(),q2.getLeadDays());o2.markPaid();o2.updateProgress(55,"坯体已完成，正在修坯并准备素烧");o2=orders.save(o2);if(payments.findByOrderId(o2.getId()).isEmpty())payments.save(new PaymentRecord(o2.getId(),"PAY-DEMO-0002","MOCK_ALIPAY",o2.getAmount()));}
if(reviews.count()==0){var user=users.findByUsername("user").orElseThrow();var sample=works.findAll();if(sample.size()>=3){reviews.save(new WorkReview(user.getId(),user.getDisplayName(),sample.get(0).getId(),5,"实物质感很好，手作纹理很自然，沟通也很顺畅。"));reviews.save(new WorkReview(user.getId()+100,"林同学",sample.get(1).getId(),4,"包装细致，木材颜色和页面展示基本一致。"));reviews.save(new WorkReview(user.getId()+101,"陈女士",sample.get(2).getId(),5,"很适合作为礼物，定制刻字效果满意。"));}}
};}
private void ensureUser(UserRepository users,BCryptPasswordEncoder encoder,String username,String password,String email,String display,String role,String avatar){if(users.findByUsername(username).isEmpty())users.save(new UserAccount(username,encoder.encode(password),email,display,role,avatar));}
private void seedWorks(CraftWorkRepository works,Long creatorId){
Object[][] rows={
{"山雾陶杯","陶艺",168d,7,"云岫陶坊","手拉坯高温烧制，杯壁保留自然手作纹理，可定制刻字与釉色。","/crafts-photo/pottery-mug.png",286,true},
{"暮山陶瓶","陶艺",268d,10,"云岫陶坊","哑光陶土花器，轮廓简洁，适合干花与桌面陈设，可定制釉色。","/crafts-photo/pottery-vase.png",184,true},
{"胡桃木戒指盒","木作",329d,12,"木屿手作","黑胡桃木手工打磨，磁吸开合，支持刻字、内衬颜色与尺寸定制。","/crafts-photo/wood-ring-box.png",336,true},
{"榉木暖光台灯","木作",428d,15,"木屿手作","榉木灯架配暖光灯罩，手工打磨上蜡，可调整木色与灯罩材质。","/crafts-photo/wood-lamp.png",214,true},
{"植物染手提包","织染",239d,10,"青染工作室","棉麻布料植物染色并手工缝制，可根据色系、尺寸和图案进行定制。","/crafts-photo/dyed-tote.png",198,true},
{"藤编野餐篮","编织",358d,14,"藤语工房","天然藤条手工编织，适合野餐与家居收纳，可定制内衬与尺寸。","/crafts-photo/woven-basket.png",165,false},
{"银杏刺绣胸针","刺绣",96d,5,"一针一线","以银杏叶为灵感，手工盘线刺绣，可选择配色与名字缩写。","/crafts-photo/embroidery-brooch.png",154,false},
{"四季花草绣框","刺绣",298d,16,"一针一线","以四季植物为主题的手工绣框，可按纪念日期设计花材组合。","/crafts-photo/embroidery-frame.png",121,true},
{"植鞣皮手账本","皮艺",199d,8,"慢皮记","植鞣牛皮手工裁切缝制，可定制封面压印、绑带与内页规格。","/crafts-photo/leather-notebook.png",177,false},
{"复古短夹","皮艺",289d,10,"慢皮记","头层植鞣革手缝短夹，支持卡位数量、皮色与姓名压印定制。","/crafts-photo/leather-wallet.png",146,true},
{"月光银饰项链","银饰",369d,9,"山白银工","925银手工锤纹吊坠，保留轻微手作肌理，可定制长度与刻字。","/crafts-photo/silver-necklace.png",203,true},
{"山纹银戒","银饰",259d,7,"山白银工","以山脊线为灵感的925银开口戒，支持圈口、纹理与内圈刻字定制。","/crafts-photo/silver-ring.png",188,false},
{"青花茶席套装","陶艺",498d,18,"云岫陶坊","茶壶、品茗杯与茶则组合，支持器型、青花纹样和题字定制。","/crafts-photo/tea-set.png",412,true},
{"赤陶餐盘","陶艺",128d,6,"云岫陶坊","手作赤陶餐盘，保留自然边缘与火痕，每件纹理略有差异。","/crafts-photo/pottery-plate.png",98,false},
{"黑胡桃木托盘","木作",238d,11,"木屿手作","整木浅盘手工打磨，边缘圆润，可定制尺寸、刻字及把手样式。","/crafts-photo/wood-tray.png",132,false},
{"草编杯垫套组","编织",88d,5,"藤语工房","天然草绳编织六件套杯垫，适合茶席与餐桌，可定制配色。","/crafts-photo/woven-coaster.png",79,false},
{"手缝皮钥匙包","皮艺",139d,6,"慢皮记","植鞣皮手缝钥匙包，黄铜五金，可定制皮色与姓名压印。","/crafts-photo/leather-keycase.png",105,false},
{"枝叶银耳饰","银饰",219d,7,"山白银工","925银枝叶造型耳饰，手工锤纹与抛光，可选择耳钩或耳针。","/crafts-photo/silver-earrings.png",167,false}
};
for(Object[] r:rows){String title=(String)r[0];CraftWork w=works.findByTitle(title).orElse(new CraftWork(title,(String)r[1],(Double)r[2],(Integer)r[3],(String)r[4],(String)r[5],(String)r[6],(Integer)r[7],(Boolean)r[8]));if(w.getId()!=null)w.refresh((String)r[1],(Double)r[2],(Integer)r[3],(String)r[4],(String)r[5],(String)r[6],(Boolean)r[8]);w.setCreatorId(creatorId);w.setStatus(CraftWork.APPROVED);works.save(w);}
}
}