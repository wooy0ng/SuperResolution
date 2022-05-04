import React, { Component, useEffect, useRef } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import Form from "react-bootstrap/Form";
import "./App.css";
import { Container, Navbar, Button } from "react-bootstrap";
import { withCookies, Cookies } from "react-cookie";
import { instanceOf } from "prop-types";
import HiddenCropper from "react-bootstrap-image-cropper/dist/HiddenCropper";
import ReactCrop from "react-image-crop";

class App extends Component {
  constructor(props) {
    super(props);
    this.id = 0;
    this.name = "";
    this.prev_img = "";
    this.result = "";

    this.inputRef = React.createRef();
    this.downloadRef = React.createRef();
    this.downloadRefs = [];

    this.state = {
      imgs: [],
      param: 50,
      results: [],
      loading_flag: false,
    };

    // https://sirong.tistory.com/101 : 쿠키
    // https://koreanred.tistory.com/112 : 쿠키

    this.handleRemove = (id) => {
      const filtered = this.state.imgs.filter((obj) => obj.id !== id);
      console.log(filtered);

      // this.result = "";
      this.setState({ imgs: filtered });

      this.setState({
        results: [],
      });
      let results = [];
      for (let idx = 0; idx < filtered.length; idx++) {
        results.push(filtered[idx].result);
      }
      this.result = results[results.length - 1];
      this.setState({ results: results });
      console.log(this.state);
    };

    this.handleSubmit = (event) => {
      const send_data = Object.assign({}, this.state);
      if (send_data.imgs.length > 0) {
        this.setState({
          loading_flag: true,
        });
        fetch("train", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(send_data),
        })
          .then((res) => res.json())
          .then((res) => {
            console.log(res);
            this.setState({
              imgs: res.data,
              loading_flag: false,
            });

            if (this.state.results.length > 0) {
              this.setState({
                results: [],
              });
            }
            let results = [];
            for (let idx = 0; idx < this.state.imgs.length; idx++) {
              results.push(this.state.imgs[idx].result);
            }
            this.result = results[results.length - 1];
            this.setState({ results: results });
          });
      } else {
        alert("please load images!!!");
      }
    };
    this.handleRefClicked = () => {
      console.log(this.downloadRefs);
      for (let idx = 0; idx < this.state.results.length; idx++) {
        this.downloadRefs[idx].current.click();
      }
    };
    // type : blob
    this.handleCropped = (blob) => {
      let idx = -1;
      if (this.state.imgs.length > 0) {
        idx = this.state.imgs[this.state.imgs.length - 1].id;
      }

      console.log("blob : ", blob);
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      reader.onloadend = (event) => {
        console.log(event);
        let base64data = reader.result;

        this.setState({
          imgs: [
            ...this.state.imgs,
            {
              id: idx + 1,
              name: `${idx + 1}.jpeg`,
              data: base64data,
              result: "",
            },
          ],
        });
      };
    };
  }
  makeDownLoadRefs() {
    for (let idx = 0; idx < this.state.results.length; idx++) {
      this.downloadRefs[idx] = React.createRef();
    }
  }

  render() {
    this.makeDownLoadRefs();
    console.log(this.state);
    return (
      <div>
        <div className="header">
          <div className="nav-bar">
            <Navbar expand="sm" variant="light" bg="light">
              <Container>
                <Navbar.Brand href="#">Super Resolution Tool</Navbar.Brand>
              </Container>
            </Navbar>
          </div>
        </div>

        <div className="body">
          <div className="container-flex main-margin padding-top-60">
            <div className="img-container margin-left-80">
              {this.state.results.length !== 0 &&
                this.state.results.map((result, idx) => (
                  <a
                    href={result}
                    ref={this.downloadRefs[idx]}
                    download={idx}
                    hidden
                  ></a>
                ))}
              <div
                className="result-img margin-left-20 margin-top-20 margin-right-20"
                style={{
                  background: `url(${
                    this.state.imgs.length > 0 && this.result
                  }) center/cover no-repeat`,
                }}
              >
                <div className="prev-img-gallary margin-top-10 margin-left-20">
                  {this.state.imgs.map((imgs, idx) => (
                    <div className="container-block" key={idx}>
                      <img
                        src={imgs.data}
                        alt=""
                        className="prev-img-box"
                        onClick={() => {
                          if (imgs.result !== "") {
                            this.id = idx;
                            this.name = imgs.name;
                            this.prev_img = imgs.data;
                            this.result = imgs.result;
                          }

                          this.setState({ imgs: [...this.state.imgs] });
                        }}
                      />
                      <div
                        className="cancel-btn"
                        onClick={() => {
                          this.handleRemove(imgs.id);
                        }}
                      >
                        -
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="text-container padding-top-30 padding-left-70">
              <div className="">
                <p className="font-title">Super Resolution Tool</p>
              </div>
              <p>
                the super Resolution Tool uses Deep Learning to clarify,
                sharpen, and upscale
                <br />
                the photo without losing its content and defining
                characteristics.
                <br />
                Super Resolution uses Deep Learning techniques to upscale images
                in a fraction of a second.
              </p>
              <div className="container-flex flex-direction-col margin-top-50">
                <div className="padding-top-30">
                  {/* param */}
                  {/* <div className="container-flex">
                    <Form.Label> α : ({this.state.param})</Form.Label>
                  </div>
                  <Form.Range
                    tooltip="on"
                    onChange={(event) => {
                      this.setState({ param: event.target.value });
                    }}
                  /> */}
                </div>
                {/* cropping test 1 */}
                <HiddenCropper
                  triggerRef={this.inputRef}
                  onCropped={this.handleCropped}
                  inputOptions={{ mimeType: "image/jpeg" }}
                  cropOptions={{ aspect: 4 / 4, maxZoom: 10 }}
                  outputOptions={{ maxWidth: 256, maxHeight: 256 }}
                  previewOptions={{ width: 256, height: 256 }}
                  displayOptions={{ title: "Cropping" }}
                />
                {/* cropping test 2 */}
                {/* 
                <input
                  type="file"
                  ref={this.inputRef}
                  onChange={(event) => {
                    console.log("input files");
                    if (event.currentTarget.files.length > 0) {
                      const file = event.currentTarget.files[0];

                      // file read
                      const reader = new FileReader();
                      reader.readAsDataURL(file);
                      reader.onloadend = (event) => {
                        let idx = -1;
                        if (this.state.imgs.length > 0) {
                          idx = this.state.imgs[this.state.imgs.length - 1].id;
                        }

                        this.setState({
                          imgs: [
                            ...this.state.imgs,
                            {
                              id: idx + 1,
                              name: file.name,
                              data: event.target.result,
                              result: "",
                            },
                          ],
                        });
                      };
                    }
                    event.target.value = "";
                  }}
                  hidden
                /> */}

                {/* button area */}
                <div className="container-flex">
                  <Button
                    className="btn-size"
                    variant="outline-secondary"
                    onClick={() => {
                      if (this.state.imgs.length < 1) {
                        // this.inputRef.current.click();
                        this.inputRef.current.trigger();
                      } else {
                        alert("이전 사진을 제거하고 넣어주세요.");
                      }
                    }}
                  >
                    Load image
                  </Button>
                  <Button
                    className="btn-size margin-left-20"
                    variant="outline-secondary"
                    disabled={this.state.imgs.length === 0}
                    onClick={() => {
                      this.result = "";
                      this.setState({
                        imgs: [],
                        results: [],
                      });
                      console.log(this.state);
                    }}
                  >
                    Clear
                  </Button>
                  {!this.state.loading_flag ? (
                    <Button
                      className="btn-size margin-left-20"
                      disabled={this.state.imgs.length === 0}
                      variant="outline-primary"
                      onClick={() => {
                        this.handleSubmit();
                      }}
                    >
                      Get SR image
                    </Button>
                  ) : (
                    <button
                      class="btn-size margin-left-20 btn btn-primary"
                      type="button"
                      disabled
                    >
                      <span
                        class="spinner-border spinner-border-sm"
                        role="status"
                        aria-hidden="true"
                      ></span>
                      &nbsp; Loading...
                    </button>
                  )}

                  <Button
                    className="btn-size margin-left-20"
                    variant="outline-primary"
                    disabled={this.state.results.length === 0}
                    onClick={() => {
                      this.handleRefClicked();
                    }}
                  >
                    Download
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default withCookies(App);
