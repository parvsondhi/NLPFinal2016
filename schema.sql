drop table if exists customers;
create table customers (
  customer_id integer primary key,
  company text not null,
  email text not null
);
