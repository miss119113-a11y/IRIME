package com.zhou.craft.model;
import jakarta.persistence.*;
import java.time.LocalDateTime;
@Entity @Table(name="craft_work")
public class CraftWork {
public static final String PENDING="PENDING",APPROVED="APPROVED",REJECTED="REJECTED";
@Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
private Long creatorId;private String title;private String category;private double price;private int leadDays;private String creatorName;
@Column(length=1000) private String summary;@Column(length=1000) private String image;private int likes;private boolean featured;@Column(length=20) private String status=APPROVED;private LocalDateTime createdAt;private LocalDateTime updatedAt;
public CraftWork(){}
public CraftWork(String title,String category,double price,int leadDays,String creatorName,String summary,String image,int likes,boolean featured){this.title=title;this.category=category;this.price=price;this.leadDays=leadDays;this.creatorName=creatorName;this.summary=summary;this.image=image;this.likes=likes;this.featured=featured;this.status=APPROVED;}
@PrePersist void onCreate(){createdAt=LocalDateTime.now();updatedAt=createdAt;if(status==null)status=APPROVED;} @PreUpdate void onUpdate(){updatedAt=LocalDateTime.now();}
public Long getId(){return id;}public Long getCreatorId(){return creatorId;}public String getTitle(){return title;}public String getCategory(){return category;}public double getPrice(){return price;}public int getLeadDays(){return leadDays;}public String getCreatorName(){return creatorName;}public String getSummary(){return summary;}public String getImage(){return image;}public int getLikes(){return likes;}public boolean isFeatured(){return featured;}public String getStatus(){return status;}public LocalDateTime getCreatedAt(){return createdAt;}public LocalDateTime getUpdatedAt(){return updatedAt;}
public void setCreatorId(Long v){creatorId=v;}public void setStatus(String v){status=v;}public void setFeatured(boolean v){featured=v;}
public void refresh(String category,double price,int leadDays,String creatorName,String summary,String image,boolean featured){this.category=category;this.price=price;this.leadDays=leadDays;this.creatorName=creatorName;this.summary=summary;this.image=image;this.featured=featured;}
public void update(String title,String category,double price,int leadDays,String summary,String image){this.title=title;this.category=category;this.price=price;this.leadDays=leadDays;this.summary=summary;if(image!=null&&!image.isBlank())this.image=image;}
public void addLike(){likes++;}public void removeLike(){if(likes>0)likes--;}
}