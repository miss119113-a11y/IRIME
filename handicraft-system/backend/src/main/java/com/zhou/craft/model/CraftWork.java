package com.zhou.craft.model;
import jakarta.persistence.*;
@Entity @Table(name="craft_work")
public class CraftWork{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;
private String title;private String category;private double price;private int leadDays;private String creatorName;
@Column(length=1000)private String summary;private String image;private int likes;private boolean featured;
public CraftWork(){}
public CraftWork(String title,String category,double price,int leadDays,String creatorName,String summary,String image,int likes,boolean featured){this.title=title;this.category=category;this.price=price;this.leadDays=leadDays;this.creatorName=creatorName;this.summary=summary;this.image=image;this.likes=likes;this.featured=featured;}
public Long getId(){return id;}public String getTitle(){return title;}public String getCategory(){return category;}public double getPrice(){return price;}public int getLeadDays(){return leadDays;}public String getCreatorName(){return creatorName;}public String getSummary(){return summary;}public String getImage(){return image;}public int getLikes(){return likes;}public boolean isFeatured(){return featured;}
public void refresh(String category,double price,int leadDays,String creatorName,String summary,String image,boolean featured){this.category=category;this.price=price;this.leadDays=leadDays;this.creatorName=creatorName;this.summary=summary;this.image=image;this.featured=featured;}
public void addLike(){likes++;}public void removeLike(){if(likes>0)likes--;}
}