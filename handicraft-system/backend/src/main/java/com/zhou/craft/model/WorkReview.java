package com.zhou.craft.model;
import jakarta.persistence.*;import java.time.LocalDateTime;
@Entity @Table(name="work_review",uniqueConstraints=@UniqueConstraint(columnNames={"userId","workId"}))
public class WorkReview{
@Id @GeneratedValue(strategy=GenerationType.IDENTITY) private Long id;private Long userId;private String userName;private Long workId;private int rating;@Column(length=1000)private String content;private LocalDateTime createdAt;private LocalDateTime updatedAt;
public WorkReview(){} public WorkReview(Long userId,String userName,Long workId,int rating,String content){this.userId=userId;this.userName=userName;this.workId=workId;this.rating=rating;this.content=content;}@PrePersist void c(){createdAt=LocalDateTime.now();updatedAt=createdAt;}@PreUpdate void u(){updatedAt=LocalDateTime.now();}
public Long getId(){return id;}public Long getUserId(){return userId;}public String getUserName(){return userName;}public Long getWorkId(){return workId;}public int getRating(){return rating;}public String getContent(){return content;}public LocalDateTime getCreatedAt(){return createdAt;}public void update(int rating,String content){this.rating=rating;this.content=content;}
}